#!/usr/bin/env python3
"""
Convert MovieLens datasets to our movie database format and save as JSON.

Usage:
    python scripts/convert_movielens.py          # Convert 100K (default)
    python scripts/convert_movielens.py --version 100k
    python scripts/convert_movielens.py --version 1m
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import MovieDatabase


def main():
    parser = argparse.ArgumentParser(description="Convert MovieLens datasets")
    parser.add_argument(
        "--version",
        choices=["100k", "1m"],
        default="100k",
        help="MovieLens version to convert (default: 100k)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="MovieLens data directory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output JSON file (auto-named if not specified)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N movies from dataset (optional)",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=None,
        help="Filter movies with minimum rating (1-10)",
    )

    args = parser.parse_args()

    # Determine paths
    if args.data_dir is None:
        args.data_dir = f"./data/movielens/ml-{args.version}k"

    if args.output_file is None:
        args.output_file = f"./data/movielens/movielens_{args.version}k.json"

    print("\n" + "=" * 70)
    print("🎬 MovieLens Dataset Converter")
    print("=" * 70)
    print(f"\nVersion: MovieLens {args.version.upper()}K")
    print(f"Source: {Path(args.data_dir).absolute()}")
    print(f"Output: {Path(args.output_file).absolute()}")

    try:
        # Load and convert dataset
        print("\n📂 Loading dataset...")
        if args.version == "100k":
            movies = MovieDatabase.from_movielens_100k(args.data_dir)
        else:  # 1m
            movies = MovieDatabase.from_movielens_1m(args.data_dir)

        print(f"   Loaded {len(movies)} movies")

        # Apply filters
        if args.min_rating is not None:
            print(f"\n🔍 Filtering by minimum rating: {args.min_rating}...")
            movies = MovieDatabase.filter_by_rating(movies, min_rating=args.min_rating)
            print(f"   After filter: {len(movies)} movies")

        if args.sample is not None:
            print(f"\n📊 Sampling {args.sample} movies...")
            movies = MovieDatabase.sample_movies(movies, args.sample, seed=42)
            print(f"   Sampled: {len(movies)} movies")

        # Save to JSON
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving to {output_path}...")
        MovieDatabase.save_json(movies, str(output_path))

        print("\n" + "=" * 70)
        print("✅ Conversion completed successfully!")
        print(f"\n📈 Dataset Statistics:")
        print(f"   Total movies: {len(movies)}")

        ratings = [m['rating'] for m in movies.values()]
        years = [m['year'] for m in movies.values()]
        print(f"   Avg rating: {sum(ratings)/len(ratings):.1f}")
        print(f"   Year range: {min(years)}-{max(years)}")

        print(f"\n💡 Usage in training:")
        print(f"   python training.py --movie-database {output_path}")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 First, download the dataset:")
        print(f"   python scripts/download_movielens.py --version {args.version}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
