import argparse
from datetime import datetime

from .data_fetcher import FetchConfig, fetch_least_cloud_sentinel2, parse_bbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch least-cloud Sentinel-2 composite.")
    parser.add_argument("--bbox", required=True, help='JSON bbox, e.g. "[88.5,21.5,89.5,22.8]"')
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    config = FetchConfig(
        bbox=parse_bbox(args.bbox),
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d").date(),
    )
    ds = fetch_least_cloud_sentinel2(config)
    print(ds)


if __name__ == "__main__":
    main()

