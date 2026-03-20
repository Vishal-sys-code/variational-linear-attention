import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Orchestrator for VLA")
    parser.add_argument("--lra_tasks", nargs="+", default=["listops", "cqa", "clutrr"], help="LRA-style tasks to run")
    parser.add_argument("--models", nargs="+", default=["linear_transformer", "deltanet", "vla"], help="Models to benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--skip_synthetic", action="store_true", help="Skip the synthetic memory tasks")
    args = parser.parse_args()

    print("="*60)
    print("      VLA UNIFIED BENCHMARK SUITE")
    print("="*60)

    # 1. Run the LRA-style tasks (Reasoning + Comprehension)
    print("\n[PHASE 1] Executing LRA-style Reasoning and Structure Tasks")
    print(f"Tasks: {args.lra_tasks}")
    
    cmd_lra = [
        sys.executable, "scripts/run_lra.py",
        "--tasks", *args.lra_tasks,
        "--models", *args.models,
        "--seeds", *[str(s) for s in args.seeds]
    ]
    
    res_lra = subprocess.run(cmd_lra)
    if res_lra.returncode != 0:
        print("!!! [ERROR] Phase 1 (LRA tasks) encountered an error.")
        sys.exit(1)
        
    print("[SUCCESS] Phase 1 completed.")

    # 2. Run the Synthetic tasks (Memory)
    if not args.skip_synthetic:
        print("\n[PHASE 2] Executing Synthetic Memory Tasks")
        cmd_synth = [sys.executable, "scripts/run_all_synthetic.py"]
        res_synth = subprocess.run(cmd_synth)
        if res_synth.returncode != 0:
            print("!!! [ERROR] Phase 2 (Synthetic tasks) encountered an error.")
            sys.exit(1)
        print("[SUCCESS] Phase 2 completed.")
    else:
        print("\n[PHASE 2] Skipped Synthetic Memory Tasks.")

    print("\n" + "="*60)
    print(" BENCHMARK SUITE COMPLETE! View results in Weights & Biases")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
