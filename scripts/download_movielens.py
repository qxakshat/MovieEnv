#!/usr/bin/env python3
"""
Download MovieLens datasets.

Usage:
    python scripts/download_movielens.py          # Download 100K (default)
    python scripts/download_movielens.py --version 100k
    python scripts/download_movielens.py --version 1m
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import MovieLensDownloader


def main():
    parser = argparse.ArgumentParser(description="Download MovieLens datasets")
    parser.add_argument(
        "--version",
        choices=["100k", "1m"],
        default="100k",
        help="MovieLens version to download (default: 100k)",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/movielens",
        help="Output directory for dataset (default: ./data/movielens)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎬 MovieLens Dataset Downloader")
    print("=" * 70)

    if args.version == "100k":
        print("\nDownloading MovieLens 100K (1682 movies, 100K ratings)...")
        success = MovieLensDownloader.download_100k(args.output_dir)
    else:  # 1m
        print("\nDownloading MovieLens 1M (3883 movies, 1M ratings)...")
        success = MovieLensDownloader.download_1m(args.output_dir)

    print("\n" + "=" * 70)
    if success:
        print("✅ Download completed successfully!")
        print(f"\n📊 Dataset location: {Path(args.output_dir).absolute()}")
        print("\n💡 Next steps:")
        print(f"   1. Convert dataset: python scripts/convert_movielens.py --version {args.version}")
        print(f"   2. Train with dataset: python training.py --dataset movielens_{args.version}")
    else:
        print("❌ Download failed")
        sys.exit(1)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
