#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
from trafilatura import feeds, extract
from trafilatura.downloads import (
    add_to_compressed_dict,
    buffered_downloads,
    load_download_buffer,
)

from .rss_feeds import CONTAMINATION_RSS

unwanted_keywords = [
    "video",
    "galeri",
    "foto",
    "hava-durumu" "havadurumu",
    "astroloji",
    "burcu",
]


def preprocess(text):
    if any(keyword in text for keyword in unwanted_keywords):
        return ""
    lines = text.split("\n")
    lines = [l for l in lines if l.strip() != "" and (not l.startswith("-"))]
    return "\n".join(lines)


def filter_keywords(urls):
    filtered_urls = []
    for url in urls:
        if not any(keyword in url for keyword in unwanted_keywords) and len(url) > 55:
            filtered_urls.append(url)
    return filtered_urls


def get_updated_rss_feeds(rss_urls):
    rss_feeds = []

    for rss_url in tqdm(rss_urls, desc="Getting RSS feeds"):
        try:
            rss_feed = feeds.find_feed_urls(rss_url, target_lang="tr")
        except:
            print(f"Falied to get rss feed from {rss_url}")
            rss_feed = []
        rss_feeds.extend(rss_feed)
    rss_feeds = filter_keywords(rss_feeds)
    return rss_feeds


def get_contamination_test_data():
    urls = set(get_updated_rss_feeds(CONTAMINATION_RSS))
    return sorted(urls)


def download_contamination_test_data(urls):

    results = {}
    threads = 32
    url_store = add_to_compressed_dict(urls)

    # processing loop
    while url_store.done is False:
        bufferlist, url_store = load_download_buffer(url_store, sleep_time=0.1)
        for url, result in tqdm(buffered_downloads(bufferlist, threads), total=len(bufferlist)):
            try: 
                text = extract(result, include_comments=False, include_tables=False)
                if text is not None:
                    text = preprocess(text)
                    if len(text) > 500:
                        results[url] = text
            except:
                print(f"Failed to fetch/extract text from {url}")

    return results
