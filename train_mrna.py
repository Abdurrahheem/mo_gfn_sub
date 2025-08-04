import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random
import wandb
from mo_gfn.torch_seq_moo.algorithms.mogfn import MOGFN

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # --- Initialize Custom Components ---
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    # --- Set model parameters from tokenizer ---
    cfg.algorithm.model.vocab_size = len(tokenizer.full_vocab)
    cfg.algorithm.model.num_actions = len(tokenizer.non_special_vocab) + 1

    # 2. Instantiate the MRNADesignTask
    #    Hydra automatically loads the correct task config based on the
    #    command-line override (e.g., `task=mrna`).
    task = hydra.utils.instantiate(cfg.task, tokenizer=tokenizer)

    # 3. Instantiate the MOGFN Algorithm
    #    The main config 'cfg' contains algorithm and model parameters.
    algorithm = MOGFN(
        cfg=cfg.algorithm, task=task, tokenizer=tokenizer, task_cfg=cfg.task)

    # Initialize wandb for logging if configured.
    # The check should be against `wandb_mode`, not `use_wandb`.
    # if cfg.wandb_mode == 'online':
    #     wandb.init(
    #         project=cfg.wandb_project,
    #         config=OmegaConf.to_container(cfg, resolve=True),
    #         name=cfg.run_name
    #     )

    # print(f"Protein Sequence: {task.protein_seq}")
    # print(f"Sequence Length: {len(task.protein_seq)} codons")

    # print("\n\n\n\t\t\t Before Optimization \n\n\n")
    # --- Run Optimization ---
    results = algorithm.optimize(task)

    # print("\nOptimization finished.")
    # print("Final results:", results)

    if cfg.wandb_mode == 'online':
        wandb.finish()

if __name__ == "__main__":
    main() 