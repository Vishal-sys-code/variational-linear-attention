import subprocess
import sys
import os

def main():
    scripts = [
        "tests/test_vla_math_regression.py",
        "scripts/run_copy_task.py",
        "scripts/run_delayed_recall.py",
        "scripts/run_associative_recall.py"
    ]
    
    # Ensure working directory is project root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    for script in scripts:
        print(f"=== Running {script} ===")
        ret = subprocess.call([sys.executable, script])
        if ret != 0:
            print(f"!!! Error in {script}. Stopping.")
            break
            
    print("=== All synthetic tasks finished ===")

if __name__ == "__main__":
    main()
