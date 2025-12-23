#!/usr/bin/env python
import os
import re

def main():
    """Extract RUN_ID from mlflow run log"""
    log_file = 'run.log'
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Cari RUN_ID dari output MLflow
        match = re.search(r'Run ID:\s+([a-f0-9]+)', content)
        if match:
            run_id = match.group(1)
            print(f"Found RUN_ID: {run_id}")
            
            # Simpan ke environment variable untuk GitHub Actions
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write(f'RUN_ID={run_id}\n')
            
            return run_id
    
    print("❌ Could not find RUN_ID in log")
    return None

if __name__ == "__main__":
    main()