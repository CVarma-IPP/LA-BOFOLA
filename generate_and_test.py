#!/usr/bin/env python3
import os
import time
import random
import pandas as pd
import argparse

def generate(folder, n):
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        # Create a timestamp-based filename seed
        timestamp = int(time.time() * 1000)
        # Randomly generate metrics
        score = round(random.uniform(0, 100), 4)
        cost = round(random.uniform(0, 50), 4)
        counts = round(random.uniform(0, 20), 4)

        # Write .txt file with "score,cost,counts"
        txt_fn = os.path.join(folder, f"espec_{timestamp}_{i:03d}.txt")
        with open(txt_fn, 'w') as f:
            f.write(f"{score},{cost},{counts}")

        # Write .csv file with columns Score and Charge
        csv_fn = os.path.join(folder, f"espec_{timestamp}_{i:03d}.csv")
        df = pd.DataFrame([{'Score': score, 'Charge': counts, 'Cost': cost}])
        df.to_csv(csv_fn, index=False)

        # Short pause so filenames differ
        time.sleep(0.05)

    print(f"Generated {n} .txt and .csv files in '{folder}'")

def main():
    parser = argparse.ArgumentParser(
        description="Generate test E-Spec data files for collect_metrics() testing"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Output folder where test files will be written"
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=30,
        help="Number of .txt/.csv file pairs to create"
    )
    args = parser.parse_args()
    folder = args.folder
    if folder is None:
        folder = input("Enter output folder path: ").strip()
        if not folder:
            print("No folder provided. Exiting.")
            return
    generate(folder, args.count)

if __name__ == "__main__":
    main()
