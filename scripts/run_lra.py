import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Orchestrator for 45 LRA Runs")
    parser.add_argument("--tasks", nargs="+", default=["listops", "retrieval", "pathfinder"])
    parser.add_argument("--models", nargs="+", default=["linear_transformer", "deltanet", "vla"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--parallel-seeds", action="store_true", default=False, help="Run seeds in parallel for each model/task pair")
    
    args = parser.parse_args()
    
    total_runs = len(args.tasks) * len(args.models) * len(args.seeds)
    print(f"Starting execution of {total_runs} LRA training runs...")
    
    run_idx = 1
    
    for task in args.tasks:
        for model in args.models:
            print(f"\n[{run_idx}/{total_runs}] Starting Task: {task} | Model: {model} with {len(args.seeds)} seeds...")
            
            if args.parallel_seeds:
                processes = []
                for seed in args.seeds:
                    cmd = [
                        sys.executable, "scripts/train_lra_worker.py",
                        "--task", task,
                        "--model", model,
                        "--seed", str(seed)
                    ]
                    print(f"  Launching Seed {seed}: {' '.join(cmd)}")
                    p = subprocess.Popen(cmd)
                    processes.append((seed, p))
                    
                # Wait for all seeds of this model to finish before next model
                for seed, p in processes:
                    p.wait()
                    if p.returncode != 0:
                        print(f"  [ERROR] Run for {task}-{model}-seed{seed} failed with code {p.returncode}")
                    else:
                        print(f"  [SUCCESS] {task}-{model}-seed{seed} completed.")
                
                run_idx += len(args.seeds)
            else:
                for seed in args.seeds:
                    cmd = [
                        sys.executable, "scripts/train_lra_worker.py",
                        "--task", task,
                        "--model", model,
                        "--seed", str(seed)
                    ]
                    print(f"  Executing Seed {seed}: {' '.join(cmd)}")
                    res = subprocess.run(cmd)
                    if res.returncode != 0:
                        print(f"  [ERROR] {task}-{model}-seed{seed} failed!")
                    run_idx += 1
                        
    print("\nAll LRA runs complete. Generate plots via scripts/analyze_lra.py")

if __name__ == "__main__":
    main()
