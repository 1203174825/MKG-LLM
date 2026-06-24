"""
MKG-LLM: Two-Stage MKG-Constrained LLM for Flight Departure Delay Prediction.
"""
import os
import sys
import argparse
import torch
import random
import numpy as np
from src.utils.config import CONFIG


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Aeolus_V2: KG-Constrained LLM for Flight Delay Prediction"
    )
    parser.add_argument("--stage", type=str, default="1",
                        choices=["1", "2", "all"])
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to Stage 1 checkpoint for Stage 2 training")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=CONFIG.train.fp16)
    args = parser.parse_args()

    CONFIG.train.fp16 = args.fp16
    os.makedirs(CONFIG.paths.output_dir, exist_ok=True)
    set_seed(CONFIG.train.seed)

    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPU(s):")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    else:
        print("WARNING: CUDA not available, falling back to CPU")

    if args.eval:
        print("\n" + "=" * 60)
        print("EVALUATION on Test Set (2017 January)")
        print("=" * 60)
        ckpt_path = os.path.join(CONFIG.paths.output_dir, "stage1_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Stage 1 checkpoint not found at {ckpt_path}")
            print("Please run Stage 1 training first.")
            sys.exit(1)

        from src.train.test_stage1 import test_stage1
        test_stage1()

    else:
        if args.stage in ("1", "all"):
            print("\n" + "=" * 60)
            print("STAGE 1: KG Encoder Pre-Training")
            print("Data: 2016 full year (~4.64M flights, 366 daily KG snapshots)")
            print("=" * 60)
            from src.train.train_stage1 import train_stage1
            stage1_model = train_stage1()
            ckpt_path = os.path.join(CONFIG.paths.output_dir, "stage1_best.pt")
            print(f"Stage 1 complete. Checkpoint: {ckpt_path}")
        else:
            stage1_model = None

        if args.stage in ("2", "all"):
            print("\n" + "=" * 60)
            print("STAGE 2: KG-Constrained LLM Fine-Tuning (Qwen2-1.5B + LoRA)")
            print("Data: 2016 full year, Validation: 2017.3-12, Test: 2017.1")
            print("Hardware: 2x RTX 3090 (DeepSpeed ZeRO-2)")
            print("=" * 60)

            ckpt = args.resume or os.path.join(
                CONFIG.paths.output_dir, "stage1_best.pt"
            )
            if not os.path.exists(ckpt):
                print(f"WARNING: Stage 1 checkpoint not found at {ckpt}")
                print("R-GCN will be randomly initialized. Performance will suffer.")

            from src.train.train_stage2 import train_stage2
            train_stage2(stage1_checkpoint=ckpt)

    print("\nDone.")


if __name__ == "__main__":
    main()
