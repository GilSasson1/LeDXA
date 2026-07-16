#!/usr/bin/env python3
"""
Training Job Manager
Monitors training progress and automatically resubmits jobs until completion.
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path

class TrainingJobManager:
    def __init__(
        self,
        checkpoint_path: str,
        bash_script: str,
        total_epochs: int,
        run_name: str,
        checkpoints_dir: str,
        check_interval: int = 60,
        max_consecutive_failures: int = 3,
        quiet: bool = False,
        existing_job_id: str = None,
    ):
        """
        Args:
            checkpoint_path: Path to checkpoint file (for reference only)
            bash_script: Path to the bash script to submit
            total_epochs: Total number of epochs to train
            run_name: Run name used for checkpoint naming
            checkpoints_dir: Directory containing checkpoints
            check_interval: Seconds between checks
            max_consecutive_failures: Max job failures before stopping
            quiet: Only print on state changes, not every check
            existing_job_id: If provided, monitor this job instead of submitting new one
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.bash_script = Path(bash_script)
        self.total_epochs = total_epochs
        self.run_name = run_name
        self.checkpoints_dir = checkpoints_dir
        self.check_interval = check_interval
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.job_ids = []
        self.quiet = quiet
        self.last_printed_status = None
        self.existing_job_id = existing_job_id
        
    def get_current_epoch(self) -> int:
        """Placeholder for compatibility. Always returns 0 (no checkpoint loading)."""
        return 0
    
    def submit_job(self) -> str:
        """Submit job and return job ID."""
        try:
            result = subprocess.run(
                [
                    'sbatch',
                    '--export', f'ALL,RUN_NAME={self.run_name},CHECKPOINTS_DIR={self.checkpoints_dir},EPOCHS={self.total_epochs}',
                    str(self.bash_script)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            # Extract job ID from "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            self.job_ids.append(job_id)
            self.consecutive_failures = 0
            return job_id
        except subprocess.CalledProcessError as e:
            self.consecutive_failures += 1
            print(f"❌ Failed to submit job (attempt {self.consecutive_failures}): {e.stderr}")
            return None
    
    def get_job_status(self, job_id: str) -> str:
        """Get job status from SLURM."""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h'],
                capture_output=True,
                text=True,
            )
            if result.stdout:
                # Extract status (4th column)
                status = result.stdout.split()[4]
                return status
            return "NOT_FOUND"
        except Exception as e:
            print(f"⚠️  Error checking job status: {e}")
            return "ERROR"
    
    def print_status(self, epoch: int, job_id: str = None, force: bool = False):
        """Print current training status (skip in quiet mode unless force=True)."""
        if self.quiet and not force:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress = f"{epoch}/{self.total_epochs}"
        percent = (epoch / self.total_epochs) * 100
        
        print(f"[{timestamp}] 📊 Progress: {progress} ({percent:.1f}%)", end="")
        
        if job_id:
            status = self.get_job_status(job_id)
            print(f" | Job {job_id}: {status}", end="")
        
        print()
    
    def run(self):
        """Main monitoring loop."""
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║ Training Job Manager - Job Monitor                  ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        if self.existing_job_id:
            print(f"║ Monitoring existing job: {self.existing_job_id}     ║")
        else:
            print(f"║ Run: {self.run_name}")
            print(f"║ Script: {self.bash_script}")
        print(f"║ Check Interval: {self.check_interval}s")
        print("╚════════════════════════════════════════════════════════════════╝")
        print()
        
        # Monitoring loop - job status only
        if self.existing_job_id:
            job_id = self.existing_job_id
            print(f"🔍 Monitoring existing job {job_id}...")
        else:
            print(f"🚀 Submitting initial job...")
            job_id = self.submit_job()
            
            if not job_id:
                print("❌ Failed to submit initial job")
                return False
            
            print(f"✅ Submitted job {job_id}")
        
        print()
        
        while True:
            status = self.get_job_status(job_id)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Only print if status changed
            if status != self.last_printed_status:
                print(f"[{timestamp}] Job {job_id}: {status}")
                self.last_printed_status = status
            
            if status == "NOT_FOUND":
                print("✅ Job finished. SLURM auto-resume wrapper handles continuation.")
                return True
            elif status in ["FAILED", "CANCELLED"]:
                print(f"❌ Job {status}")
                return False
            
            # Wait before next check
            time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor and auto-resume training jobs"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name used for checkpoint naming"
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="/data/hpp_labdata/Analyses/gilsa/checkpoints/lejepa_dexa/",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--script",
        default="multi_gpu_bash_autoresume.sh",
        help="Path to bash script to submit"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Total epochs to train"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Check interval in seconds"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print on status changes, not at every check interval"
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Monitor an existing job ID instead of submitting a new one"
    )

    
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if not args.run_name:
            parser.error("Provide either --checkpoint or --run-name")
        checkpoint_path = str(Path(args.checkpoints_dir) / f"{args.run_name}.pth")

    effective_run_name = args.run_name if args.run_name else Path(checkpoint_path).stem

    manager = TrainingJobManager(
        checkpoint_path=checkpoint_path,
        bash_script=args.script,
        total_epochs=args.epochs,
        run_name=effective_run_name,
        checkpoints_dir=args.checkpoints_dir,
        check_interval=args.interval,
        quiet=args.quiet,
        existing_job_id=args.job_id,
    )
    
    success = manager.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
