"""Scrape international tests involving a Six Nations side and append them locally."""

import argparse
from itertools import permutations
from datetime import date
from pathlib import Path

from ..data import load_dataset
from ..data.load import DATA_DIR
from .batch import scrape_urls
from .international import discover_international_urls
from .merge import merge_scrapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DATA_DIR)
    parser.add_argument("--batch", type=Path, default=Path("data/international_additions"))
    parser.add_argument("--output", type=Path, default=Path("data/rebuild_with_internationals"))
    parser.add_argument("--years", nargs="+", type=int, metavar="YEAR",
                        help="Calendar years to check (default: 2021 through the current year)")
    parser.add_argument("--reuse-batch", action="store_true",
                        help="Merge a saved scrape batch without network requests")
    args = parser.parse_args()
    paths = [path.resolve() for path in (args.base, args.batch, args.output)]
    if any(left.is_relative_to(right) for left, right in permutations(paths, 2)):
        parser.error("--base, --batch and --output must be separate directories")
    if args.output.exists():
        parser.error(f"{args.output} already exists. Choose a new --output directory.")

    if not args.reuse_batch:
        if args.batch.exists():
            parser.error(f"{args.batch} already exists. Use --reuse-batch or choose a new path.")
        years = sorted(set(args.years or range(2021, date.today().year + 1)))
        base_data = load_dataset(args.base)
        pending, unresolved = discover_international_urls(years, base_data.matches)
        print(f"{len(pending):,} new match URLs. {len(unresolved):,} unresolved calendar links.",
              flush=True)
        failures = scrape_urls(pending, args.batch)
        pending.to_parquet(args.batch / "urls.parquet")
        unresolved.to_parquet(args.batch / "unresolved_urls.parquet", index=False)
        print(f"{len(failures):,} scrape failures. Reports saved in {args.batch}.")

    added = merge_scrapes(args.base, args.batch, args.output)
    print(f"Added {added:,} matches to {args.output}.")


if __name__ == "__main__":
    main()
