"""Discover RugbyPass fixtures and scrape them into a local dataset."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ..data.load import DATA_DIR
from .batch import scrape_urls
from .fixtures import discover_urls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/scraping/urls.parquet"),
                        help="Fixture URL table (default: %(default)s)")
    parser.add_argument("--reuse-manifest", action="store_true",
                        help="Read the saved URL table and skip fixture discovery")
    parser.add_argument("--output", type=Path, default=DATA_DIR,
                        help="New dataset directory (default: %(default)s)")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"{args.output} already exists. Choose a new --output directory.")
    if args.manifest.resolve().is_relative_to(args.output.resolve()):
        parser.error("Choose a manifest path outside the new output directory.")

    if args.reuse_manifest:
        if not args.manifest.is_file():
            parser.error(f"Cannot find manifest {args.manifest}")
        urls = pd.read_parquet(args.manifest)
    else:
        print("Discovering fixtures...", flush=True)
        urls = discover_urls()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.manifest.parent) as tmp:
            temporary = Path(tmp) / "urls.parquet"
            urls.to_parquet(temporary)
            temporary.replace(args.manifest)

    print(urls.groupby(["competition", "season"]).size().to_string())
    print(f"Scraping {len(urls):,} fixtures from {args.manifest}...", flush=True)
    failures = scrape_urls(urls, args.output)
    print(f"Accepted {len(urls) - len(failures):,} matches. Rejected {len(failures):,}.")
    print(f"Output directory: {args.output}")
    if not failures.empty:
        print(f"See {args.output / 'failures.parquet'} for rejected URLs and reasons.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
