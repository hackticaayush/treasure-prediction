"""
fetcher.py  –  Keeps round_data.json updated by polling the StarMaker API.
Run automatically by launcher.py (or standalone: python fetcher.py)
"""

import requests
import json
import time
import os

BASE_URL = "https://m.starmakerstudios.com/go-v1/ssc/2711/records?start_round="
HEADERS = {
    'User-Agent': "sm/9.9.4/Android/13/google play/d48399ffafa2d343/wifi/en-IN/SM-M325F/10977524107285207///India",
    'Accept': "application/json, text/plain, */*",
    'Accept-Encoding': "gzip, deflate, br, zstd",
    'sec-ch-ua-platform': '"Android"',
    'sec-ch-ua': '"Not:A-Brand";v="99", "Android WebView";v="145", "Chromium";v="145"',
    'sec-ch-ua-mobile': "?1",
    'x-requested-with': "com.starmakerinteractive.starmaker",
    'sec-fetch-site': "same-origin",
    'sec-fetch-mode': "cors",
    'sec-fetch-dest': "empty",
    'referer': "https://m.starmakerstudios.com/v/rhapsody-music/history?promotion_id=2711&showBar=1&showNavigation=true",
    'accept-language': "en-IN,en-US;q=0.9,en;q=0.8",
    'priority': "u=1, i",
    'Cookie': "PHPSESSID=pd6mapbqfhbk3e7argj51uh1ts; X-Rce-Type-11=yidun; _gcl_au=1.1.1850565661.1776934232; _ga=GA1.1.459020718.1776934232; oauth_token=94le54aFnKy5CrbNzo7s903FOWniysVT; X-Rce-Token-11=-ayLeuneijZRW7lrFUtrnbDRoNkveB2IEVypag==; _ga_Y5QLWEHNZ4=GS2.1.s1777212796$o7$g1$t1777213527$j27$l0$h0"
}

OUTPUT_FILE = "round_data.json"
ROUND_FIELD = "round"
MAX_RETRIES = 3
RETRY_DELAY = 2
POLL_INTERVAL = 5   # seconds between fetch cycles


def load_existing_data():
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list) or len(records) == 0:
            return [], set()
        known_rounds = {r[ROUND_FIELD] for r in records if ROUND_FIELD in r}
        return records, known_rounds
    except FileNotFoundError:
        return [], set()
    except json.JSONDecodeError:
        return [], set()


def fetch_page(url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            print(resp.text)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  [Attempt {attempt}] Error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def save_data(records):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def fetch_new_rounds():
    """
    Fetch any rounds newer than what's already in round_data.json.
    Returns the number of new records added.
    """
    existing_records, known_rounds = load_existing_data()
    existing_set = set(known_rounds)

    current_url = BASE_URL
    new_records = []
    page = 1

    while current_url:
        data = fetch_page(current_url)
        if data is None:
            print("  Fetch failed, skipping cycle.")
            break

        records = data.get("list", [])
        if not records:
            break

        fresh = []
        overlap = False
        for record in records:
            rv = record.get(ROUND_FIELD)
            if rv is not None and rv in existing_set:
                overlap = True
                break
            fresh.append(record)
            if rv is not None:
                existing_set.add(rv)

        new_records.extend(fresh)
        if overlap:
            break

        has_more = data.get("has_more", False)
        callback = data.get("callback", None)
        if not has_more or not callback:
            break

        current_url = callback
        page += 1
        time.sleep(0.3)

    if new_records:
        all_records = new_records + existing_records
        save_data(all_records)
        max_new = max(r[ROUND_FIELD] for r in new_records if ROUND_FIELD in r)
        print(f"  [Fetcher] +{len(new_records)} new round(s). Latest round: {max_new}")
    else:
        pass  # no new data, stay silent

    return len(new_records)


def run_loop():
    print("[Fetcher] Starting continuous fetch loop (every 5s)...")
    while True:
        try:
            fetch_new_rounds()
        except Exception as e:
            print(f"[Fetcher] Unexpected error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_loop()