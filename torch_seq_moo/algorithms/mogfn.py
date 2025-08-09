import hydra
import wandb
import math
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F

from torch_seq_moo.data.mrna_constants import IDX_TO_CODON

from torch_seq_moo.algorithms.base import BaseAlgorithm
from torch_seq_moo.algorithms.mogfn_utils.utils import mean_pairwise_distances, generate_simplex, thermometer, plot_pareto, pareto_frontier
from torch_seq_moo.utils import str_to_tokens, tokens_to_str
from torch_seq_moo.metrics import get_all_metrics

from torch.distributions import Categorical
from tqdm import tqdm



class MOGFN(BaseAlgorithm):
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(MOGFN, self).__init__(cfg, task, tokenizer, task_cfg)
        self.setup_vars(kwargs)
        self.init_policy()

    def setup_vars(self, kwargs):
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Task stuff
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        self.obj_dim = self.task.obj_dim
        # GFN stuff
        self.train_steps = cfg.train_steps
        self.random_action_prob = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.reward_min = cfg.reward_min
        self.therm_n_bins = cfg.therm_n_bins
        self.beta_use_therm = cfg.beta_use_therm
        self.pref_use_therm = cfg.pref_use_therm
        self.gen_clip = cfg.gen_clip
        self.sampling_temp = cfg.sampling_temp
        self.sample_beta = cfg.sample_beta
        self.beta_cond = cfg.beta_cond
        self.pref_cond = cfg.pref_cond
        self.beta_scale = cfg.beta_scale
        self.beta_shape = cfg.beta_shape
        self.pref_alpha = cfg.pref_alpha
        self.beta_max = cfg.beta_max
        self.reward_type = cfg.reward_type
        self.use_eval_pref = cfg.use_eval_pref
        self.num_pareto_points = cfg.num_pareto_points
        self.state_save_path = cfg.state_save_path
        self.pareto_freq = cfg.pareto_freq
        self.eval_prefs = self.get_eval_pref()# np.array(self.task_cfg.eval_pref)

        # Eval Stuff
        self._hv_ref = None
        self._ref_point = np.array([0] * self.obj_dim)
        self.eval_metrics = cfg.eval_metrics
        self.eval_freq = cfg.eval_freq
        self.k = cfg.k
        self.num_samples = cfg.num_samples
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.simplex = generate_simplex(self.obj_dim, cfg.simplex_bins)
        self.unnormalize_rewards = cfg.unnormalize_rewards
        # Adapt model config to task if not already set, for backward compatibility
        if not hasattr(self.cfg.model, "vocab_size"):
            self.cfg.model.vocab_size = len(self.tokenizer.full_vocab)
        if not hasattr(self.cfg.model, "num_actions"):
            self.cfg.model.num_actions = len(self.tokenizer.non_special_vocab) + 1

    def get_eval_pref(self):
        rs = np.random.RandomState(123)
        return rs.dirichlet([1] * self.obj_dim, size=5)

    def init_policy(self):
        cfg = self.cfg
        pref_dim = self.therm_n_bins * self.obj_dim if self.pref_use_therm else self.obj_dim
        beta_dim = self.therm_n_bins if self.beta_use_therm else 1
        cond_dim = pref_dim + beta_dim if self.beta_cond else pref_dim
        self.model = hydra.utils.instantiate(cfg.model, cond_dim=cond_dim, use_cond=(self.beta_cond or self.pref_cond))

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.model_params(), cfg.pi_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), cfg.z_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))


    def optimize(self, task, init_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """
        losses, rewards = [], []
        hv, r2, hsri, rs = 0., 0., 0., np.zeros(self.obj_dim)
        pb = tqdm(range(self.train_steps))
        desc_str = "Evaluation := Reward: {:.3f} HV: {:.3f} R2: {:.3f} HSRI: {:.3f} | Train := Loss: {:.3f} Rewards: {:.3f}"
        pb.set_description(desc_str.format(rs.mean(), hv, r2, hsri, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))

        for i in pb:
            loss, r = self.train_step(task, self.batch_size)
            losses.append(loss)
            rewards.append(r)
            
            if i != 0 and i % self.eval_freq == 0:
                with torch.no_grad():
                    samples, all_rews, rs, mo_metrics, topk_metrics, fig = self.evaluation(task, plot=True)
                hv, r2, hsri = mo_metrics["hypervolume"], mo_metrics["r2"], mo_metrics["hsri"]
                
                self.log(dict(
                    topk_rewards=topk_metrics[0].mean(),
                    topk_diversity=topk_metrics[1].mean(),
                    sample_r=rs.mean()
                ), commit=False)
                # print({"topk_reward_pref_{}".format(i): topk_metrics[0][i] for i in range(len(topk_metrics[0]))})
                if self.use_eval_pref:
                    self.log({"topk_reward_pref_{}".format(i): topk_metrics[0][i] for i in range(len(topk_metrics[0]))}, commit=False)
                    self.log({"topk_diversity_pref_{}".format(i): topk_metrics[1][i] for i in range(len(topk_metrics[1]))}, commit=False)
                    self.log({"sample_reward_pref_{}".format(i): rs[i] for i in range(len(rs))}, commit=False)

                self.log({key: val for key, val in mo_metrics.items()}, commit=False)

                if fig is not None:
                    self.log(dict(
                        pareto_front=fig
                    ), commit=False)
                table = wandb.Table(columns = ["Sequence", "Rewards", "Prefs"])
                if self.unnormalize_rewards:
                    all_rews *= task.score_max
                for sample, rew, pref in zip(samples, all_rews, self.simplex):
                    table.add_data(str(sample), str(rew), str(pref))
                self.log({"generated_seqs": table})
                if i % self.pareto_freq == 0:
                    new_candidates, all_rewards, r_scores, pareto_candidates, pareto_targets, prefs = self.plot_pareto(self.num_pareto_points, task)
                    self.update_state(dict(
                        topk_rewards=topk_metrics[0].mean(),
                        topk_diversity=topk_metrics[1].mean(),
                        sample_r=rs.mean(),
                        hv=mo_metrics["hypervolume"],
                        r2=mo_metrics["r2"],
                        samples=new_candidates,
                        all_rewards=all_rewards,
                        prefs=prefs
                    ))
                    self.save_state()
            # self.log(dict(
            #     train_loss=loss,
            #     train_rewards=r,
            # ))
            pb.set_description(desc_str.format(rs.mean(), hv, r2, hsri, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))
        
        # Print a sample from the trained model
        with torch.no_grad():
            if hasattr(task, "protein_seq"):
                try:
                    pref = np.ones(self.obj_dim) / self.obj_dim
                    cond_var, _ = self._get_condition_var(prefs=pref, train=False, bs=1)
                    gen_samples, _ = self.sample(1, cond_var, train=False)
                    gen_rewards = task.score(gen_samples)
                    if self.unnormalize_rewards:
                        gen_rewards = gen_rewards * task.score_max

                    print("[MOGFN] Sample after training")
                    print(f"  Protein: {task.protein_seq}")
                    print(f"  Prefs: {pref}")
                    print(f"  Seq:   {gen_samples[0]}")
                    print(f"  Rews:  {gen_rewards[0]}")
                except Exception as e:
                    print(f"[MOGFN] Failed to sample after training: {e}")

        return {
            'losses': losses,
            'train_rs': rewards,
            'hypervol_rel': hv
        }
    
    def train_step(self, task, batch_size):
        cond_var, (prefs, beta) = self._get_condition_var(train=True, bs=batch_size)
        # print("cond_var", cond_var, cond_var.shape)
        # print("prefs", prefs)
        # print("beta", beta)
        states, logprobs = self.sample(batch_size, cond_var)
        # print("states", states)
        # print("logprobs", logprobs)
    

        log_r = self.process_reward(states, prefs, task).to(self.device)

        # import sys
        # sys.exit()

        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        
        # TB Loss
        loss = (logprobs - beta * log_r).pow(2).mean()
        # print("loss", loss, "log_r", log_r)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        return loss.item(), log_r.mean()


    def sample(self, episodes, cond_var=None, train=True):
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)
        if cond_var is None:
            cond_var, _ = self._get_condition_var(train=train, bs=episodes)
        active_mask = torch.ones(episodes).bool().to(self.device)

        # Check if the task is our constrained mRNA design task
        is_constrained = hasattr(self.task, "protein_seq")
        if is_constrained:
            protein_len = len(self.task.protein_seq)
            # For our task, we generate from scratch, so x starts empty and we build it.
            # `str_to_tokens` should handle this, returning a minimal tensor.
            # print("states", states)
            x = str_to_tokens(states, self.tokenizer, use_sep=False).to(self.device).t()
            # print("x", x)
        else:
            # Original behavior for other tasks
            x = str_to_tokens(states, self.tokenizer).to(self.device).t()[:1]

        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)

        # The loop length should be guided by the protein length for the constrained case
        max_len = protein_len + 1 if is_constrained else self.max_len
        # print("max_len", max_len)

        for t in (range(max_len) if episodes > 0 else []):
            logits = self.model(x, cond_var, lens=lens, mask=None)
            # print("logits", logits.shape)
            
            # --- Start of Constrained Sampling Logic ---
            if is_constrained:
                if t < protein_len:

                    # Get valid token IDs from the task for the current position
                    valid_token_ids = self.task.get_valid_actions(t)

                    # lets print the codon for the valid token ids
                    
                    # Map token IDs to the model's action IDs.
                    # Action `i` (where i>0) corresponds to the (i-1)-th codon in the tokenizer's `non_special_vocab`.
                    # So, Action ID = Token ID - (number of special tokens) + 1
                    offset = len(self.tokenizer.special_tokens)
                    # print("offset", offset)
                    valid_actions = [tid - offset + 1 for tid in valid_token_ids if tid >= offset]
                    # print("valid actions", valid_actions)

                    
                    # Create a mask to disable all actions by default
                    action_mask = torch.full_like(logits, -1e9)
                    # Enable valid codon actions
                    action_mask[:, valid_actions] = 0
                    # Explicitly disable the stop action during generation

                    # Disable the stop action (action 0) during codon generation
                    # This prevents the model from ending the sequence prematurely
                    action_mask[:, 0] = -1e9
                    
                    # Apply the action mask to the logits to enforce constraints
                    logits = logits + action_mask

                elif t == protein_len:
                    # Force the model to select the stop action
                    action_mask = torch.full_like(logits, -1e9)
                    action_mask[:, 0] = 0 # Enable ONLY the stop action
                    logits = logits + action_mask
            else:
                 # Original unconstrained logic
                 if t <= self.min_len:
                    logits[:, 0] = -1000

            if t == 0:
                traj_logprob += self.model.Z(cond_var)
            # --- End of Constrained Sampling Logic ---

            sampling_dist = Categorical(logits=logits / self.sampling_temp)
            policy_dist = Categorical(logits=logits)
            actions = sampling_dist.sample()
            # print("actions", actions)
            log_prob = policy_dist.log_prob(actions) * active_mask
            traj_logprob += log_prob
            # print("traj_logprob", traj_logprob)
            
            # --- Start of Fixed Action-to-Token Mapping ---
            # The original `actions + 4` is hardcoded and incorrect for our tokenizer.
            # We must map actions back to token IDs correctly.
            actions_apply = torch.full_like(actions, self.tokenizer.padding_idx)
            # print("actions_apply", actions_apply)

            # Map stop action (0) to the EOS token ID
            is_stop_action = (actions == 0)
            # print("is_stop_action", is_stop_action)
            actions_apply[is_stop_action] = self.tokenizer.eos_token_id
            # print("actions_apply after stop action", actions_apply)

            # Map codon actions (>0) to their corresponding token IDs
            is_codon_action = (actions > 0)
            if torch.any(is_codon_action):
                codon_actions = actions[is_codon_action]
                # print("codon_actions", codon_actions)
                offset = len(self.tokenizer.special_tokens)
                # Token ID = Action ID + (num special tokens) - 1
                codon_token_ids = codon_actions + offset - 1
                actions_apply[is_codon_action] = codon_token_ids
                # print("actions_apply after codon actions", actions_apply)
            # --- End of Fixed Action-to-Token Mapping ---

            # For finished sequences (inactive), use padding token.
            actions_apply = torch.where(torch.logical_not(active_mask), torch.tensor(self.tokenizer.padding_idx).to(self.device), actions_apply)
            # print("actions_apply after padding", actions_apply)
            active_mask = torch.where(active_mask, actions != 0, active_mask)

            x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if active_mask.sum() == 0:
                break
            # print("--------------------------------")

        # Decode the full sequence of token IDs using our tokenizer
        states = self.tokenizer.batch_decode(x.t().cpu().tolist())
        # print("print states", states)
        # print("print traj_logprob", traj_logprob)
        return states, traj_logprob
    

    def process_reward(self, seqs, prefs, task, rewards=None, train=True):
        # import ipdb
        # ipdb.set_trace()
        if rewards is None:
            rewards = task.score(seqs)
        if self.reward_type == "convex":
            log_r = (torch.tensor(prefs) * (rewards)).sum(axis=1).clamp(min=self.reward_min).log()
        elif self.reward_type == "logconvex":
            log_r = (torch.tensor(prefs) * torch.tensor(rewards).clamp(min=self.reward_min).log()).sum(axis=1)
        elif self.reward_type == "tchebycheff":
            log_r = (torch.tensor(prefs) * torch.abs(1 - torch.tensor(rewards))).max(axis=1)[0].clamp(min=self.reward_min).log()
        return log_r.exp() if not train else log_r

    def evaluation(self, task, plot=False):
        new_candidates = []
        r_scores = [] 
        all_rewards = []
        topk_rs = []
        topk_div = []
        if self.use_eval_pref:
            for prefs in self.eval_prefs:
                cond_var, (_, beta) = self._get_condition_var(prefs=prefs, train=False, bs=self.num_samples)
                samples, _ = self.sample(self.num_samples, cond_var, train=False)
                rewards = task.score(samples)
                r = self.process_reward(samples, prefs, task, rewards=rewards, train=False)
                
                # topk metrics
                topk_r, topk_idx = torch.topk(r, self.k)
                samples = np.array(samples)
                topk_seq = samples[topk_idx].tolist()
                edit_dist = mean_pairwise_distances(topk_seq)
                topk_rs.append(topk_r.mean().item())
                topk_div.append(edit_dist)
                
                # top 1 metrics
                max_idx = r.argmax()
                new_candidates.append(samples[max_idx])
                all_rewards.append(rewards[max_idx])
                r_scores.append(r.max().item())
        else:
            for prefs in self.simplex:
                cond_var, (_, beta) = self._get_condition_var(prefs=prefs, train=False, bs=self.num_samples)
                samples, _ = self.sample(self.num_samples, cond_var, train=False)
                rewards = task.score(samples)
                r = self.process_reward(samples, prefs, task, rewards=rewards, train=False)
                
                # topk metrics
                topk_r, topk_idx = torch.topk(r, self.k)
                samples = np.array(samples)
                topk_seq = samples[topk_idx].tolist()
                edit_dist = mean_pairwise_distances(topk_seq)
                topk_rs.append(topk_r.mean().item())
                topk_div.append(edit_dist)
                
                # top 1 metrics
                max_idx = r.argmax()
                new_candidates.append(samples[max_idx])
                all_rewards.append(rewards[max_idx])
                r_scores.append(r.max().item())

        r_scores = np.array(r_scores)
        all_rewards = np.array(all_rewards)
        new_candidates = np.array(new_candidates)

        if not self.use_eval_pref:
            # filter to get current pareto front 
            pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
            
            mo_metrics = get_all_metrics(pareto_targets, self.eval_metrics, hv_ref=self._ref_point, r2_prefs=self.simplex, num_obj=self.obj_dim)
            obj_names = self.task_cfg.objectives if hasattr(self.task_cfg, "objectives") else [f"obj_{i}" for i in range(self.obj_dim)]
            fig = plot_pareto(pareto_targets, all_rewards, pareto_only=False, objective_names=obj_names) if plot else None        
        else:
            mo_metrics = {met: 0 for met in self.eval_metrics}
            fig = None
        
        return new_candidates, all_rewards, r_scores, mo_metrics, (np.array(topk_rs), np.array(topk_div)), fig

    def plot_pareto(self, num_points, task):
        prefs = np.random.dirichlet([1] * self.obj_dim, size=num_points)
        new_candidates = []
        r_scores = [] 
        all_rewards = []
        for pref in prefs:
            cond_var, (_, beta) = self._get_condition_var(prefs=pref, train=False, bs=self.num_samples)
            samples, _ = self.sample(self.num_samples, cond_var, train=False)
            rewards = task.score(samples)
            r = self.process_reward(samples, pref, task, rewards=rewards, train=False)
            
            new_candidates.extend(samples)
            all_rewards.extend(rewards)
            r_scores.extend(r.cpu())
        r_scores = np.array(r_scores)
        all_rewards = np.array(all_rewards)
        new_candidates = np.array(new_candidates)
        pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards,maximize=True)
        return new_candidates, all_rewards, r_scores, pareto_candidates, pareto_targets, prefs

    def val_step(self, batch_size):
        overall_loss = 0.
        for pref in self.simplex:
            cond_var, (prefs, beta) = self._get_condition_var(prefs=pref, train=False, bs=batch_size)
            num_batches = len(self.val_split.inputs) // self.batch_size
            losses = 0
            for i in range(num_batches):
                states = self.val_split.inputs[i * self.batch_size:(i+1) * self.batch_size]
                logprobs = self._get_log_prob(states, cond_var, batch_cond=None)
                r = self.process_reward(self.val_split.inputs[i * self.batch_size:(i+1) * self.batch_size], prefs).to(seq_logits.device)
                loss = (seq_logits - beta * r.clamp(min=self.reward_min).log()).pow(2).mean()

                losses += loss.item()
            overall_loss += (losses / num_batches)
        return overall_loss / len(self.simplex)

    def _get_log_prob(self, states, cond_var, batch_cond):
        lens = torch.tensor([len(z) + 2 for z in states]).long().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()
        mask = x.eq(self.tokenizer.padding_idx)
        logits = self.model(x, cond_var, batch_cond, mask=mask.transpose(1,0), return_all=True, lens=lens, logsoftmax=True)
        seq_logits = (logits.reshape(-1, 21)[torch.arange(x.shape[0] * x.shape[1], device=self.device), (x.reshape(-1)-4).clamp(0)].reshape(x.shape) * mask.logical_not().float()).sum(0)
        seq_logits += self.model.Z(cond_var)
        return seq_logits

    def _get_condition_var(self, prefs=None, beta=None, train=True, bs=None):
        if prefs is None:
            if not train:
                prefs = self.simplex[0]
            else:
                prefs = np.random.dirichlet(np.array(self.pref_alpha))
        if beta is None:
            if train:
                beta = float(np.random.randint(1, self.beta_max+1)) if self.beta_cond else self.sample_beta
            else:
                beta = self.sample_beta

        if self.pref_use_therm:
            prefs_enc = thermometer(torch.from_numpy(prefs), self.therm_n_bins, 0, 1) 
        else: 
            prefs_enc = torch.from_numpy(prefs)
        
        if self.beta_use_therm:
            beta_enc = thermometer(torch.from_numpy(np.array([beta])), self.therm_n_bins, 0, self.beta_max) 
        else:
            beta_enc = torch.from_numpy(np.array([beta]))
        if self.beta_cond:
            cond_var = torch.cat((prefs_enc.view(-1), beta_enc.view(-1))).float().to(self.device)
        else:
            cond_var = prefs_enc.view(-1).float().to(self.device)
        if bs:
            cond_var = torch.tile(cond_var.unsqueeze(0), (bs, 1))
        return cond_var, (prefs, beta)
