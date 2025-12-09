#!/usr/bin/env python3
"""
Cleanup script for RunPod model volume setup
This script helps free up space before model downloads
"""

import os
import shutil
from pathlib import Path
import argparse


def cleanup_temp_files(base_path):
    """Clean up temporary files that might be taking up space"""
    base_dir = Path(base_path)

    print(f"Cleaning up temporary files in {base_dir}...")

    # Remove common temporary files and directories
    temp_patterns = [
        "__pycache__",
        "*.tmp",
        "*.temp",
        ".cache",
        ".git/objects/pack/pack-*.idx",
        "tmp",
        "temp"
    ]

    cleaned_count = 0
    for pattern in temp_patterns:
        for item in base_dir.rglob(pattern):
            try:
                if item.is_file():
                    item.unlink()
                    print(f"Removed temporary file: {item}")
                    cleaned_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    print(f"Removed temporary directory: {item}")
                    cleaned_count += 1
            except Exception as e:
                print(f"Could not remove {item}: {e}")

    print(f"Cleaned up {cleaned_count} temporary items")


def check_disk_space(path):
    """Check available disk space"""
    try:
        total, used, free = shutil.disk_usage(path)
        print(f"Disk space at {path}:")
        print(f"  Total: {total // (1024**3)} GB")
        print(f"  Used: {used // (1024**3)} GB")
        print(f"  Free: {free // (1024**3)} GB")
        return free
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Cleanup script for RunPod model setup")
    parser.add_argument(
        "--path",
        type=str,
        default="/workspace",
        help="Path to clean up (default: /workspace)"
    )
    parser.add_argument(
        "--check-space",
        action="store_true",
        help="Only check disk space without cleaning"
    )

    args = parser.parse_args()

    if args.check_space:
        check_disk_space(args.path)
        return

    print("Starting cleanup process...")

    # Check disk space before cleanup
    free_before = check_disk_space(args.path)

    # Clean up temporary files
    cleanup_temp_files(args.path)

    # Check disk space after cleanup
    free_after = check_disk_space(args.path)

    if free_after > free_before:
        gained_space = (free_after - free_before) // (1024**2)  # MB
        print(f"\nGained approximately {gained_space} MB of space")


if __name__ == "__main__":
    main()