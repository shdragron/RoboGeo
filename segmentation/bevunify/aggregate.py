"""Robustness scores from per-cell result files.

    python -m bevunify.aggregate --results eval_results/gaussianlss/results [--out summary.json]

Refuses incomplete grids unless --allow-partial is given. The formulas are the same as in
detection/robustness/aggregate.py, which also reads these files.
"""
import argparse
import json
import sys

from .robustness import aggregate, format_summary, load_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="directory with the per-cell JSON files")
    parser.add_argument("--out", help="write the summary JSON here")
    parser.add_argument("--allow-partial", action="store_true", help="score incomplete grids from the cells present")
    args = parser.parse_args()

    summary, incomplete = aggregate(load_results(args.results), allow_partial=args.allow_partial)
    for message in incomplete:
        print(f"incomplete grid: {message}", file=sys.stderr)
    if incomplete and not args.allow_partial:
        sys.exit(1)
    print(format_summary(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
