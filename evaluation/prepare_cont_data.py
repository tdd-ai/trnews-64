from utils.contamination import (
    download_contamination_test_data,
    get_contamination_test_data,
)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Prepare test data for contamination test")
    parser.add_argument("--output", default="trnews.cont.raw", help="Output file")
    args = parser.parse_args()

    print(f"Getting URLs from RSS feeds...")
    urls = get_contamination_test_data()
    print(f"Downloading {len(urls)} articles...")
    results = download_contamination_test_data(urls)

    with open(args.output + ".urls", "w") as fi:
        for url in urls:
            fi.write(f"{url}\n")

    with open(args.output, "w") as fi:
        for url, result in results.items():
            fi.write(f"{result}\n\n")

    print(f"Saved {len(urls)} articles to {args.output}")
