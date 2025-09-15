# Script to scrape abstracts from conferences on ACL Anthology
# Saves the abstracts to a JSON file.
# Usage: python3 abstract_collector.py <conference_acronym>
# RTSREC001 - Rector Ratsaka

import requests
from bs4 import BeautifulSoup
import json
import time
import argparse

# command line args
parser = argparse.ArgumentParser(description="Scrape abstracts from ACL Anthology conferences.")
parser.add_argument("conference", type=str, help="Conference acronym (e.g., lrec, cl, wmt, ranlp, conll).")
args = parser.parse_args()
conference = args.conference.lower()

BASE_URL = "https://aclanthology.org"
VENUE_URL = f"{BASE_URL}/venues/{conference}/"

# Get all volume links from the venue page
res = requests.get(VENUE_URL)
soup = BeautifulSoup(res.text, "html.parser")

volume_links = soup.find_all("a", href=True)
volume_urls = sorted(set(
    BASE_URL + link["href"]
    for link in volume_links
    if link["href"].startswith("/volumes/") and link["href"].endswith("/")
))

print(f"Found {len(volume_urls)} volume URLs")

# Loop over each volume and extract abstracts
results = []
paper_id = 1

for volume_url in volume_urls:
    print(f"Scraping: {volume_url}")
    try:
        vres = requests.get(volume_url)
        vsoup = BeautifulSoup(vres.text, "html.parser")

        abstracts = vsoup.find_all("div", class_="card-body p-3 small")
        titles = vsoup.find_all("p", class_="d-sm-flex align-items-stretch")
        

        for abstract, title_block in zip(abstracts, titles):
            title_tag = title_block.find("strong")
            link_tag = title_tag.find("a") if title_tag else None

            title = title_tag.text.strip() if title_tag else "No Title"
            abstract_text = abstract.text.strip()
            url = "https://aclanthology.org" + link_tag['href'] if link_tag else "No URL"

            results.append({
                "id": paper_id,
                "title": title,
                "abstract": abstract_text,
                "url": url
            })
            paper_id += 1

        time.sleep(1) 

    except Exception as e:
        print(f"Error scraping {volume_url}: {e}")
        continue

# Save results to JSON
with open(f"{conference}_abstracts.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nFinished. Collected {len(results)} abstracts.")