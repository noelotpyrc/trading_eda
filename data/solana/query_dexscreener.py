import csv
import requests
import time
import json

input_csv = "solana_mint_volume.csv"
output_json = "solana_dexscreener_info.json"

results = []

# TEST MODE: Only process the first 10 tokens and time it
TEST_MODE = False
TEST_LIMIT = 10
start_time = time.time()

# Rate limit settings
RATE_LIMIT = 300  # requests
RATE_PERIOD = 60  # seconds
req_counter = 0
window_start = time.time()

with open(input_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for idx, row in enumerate(reader, 1):
        mint_address = row['mint_address']
        print(f"[{idx}] Querying: {mint_address}")
        url = f"https://api.dexscreener.com/latest/dex/tokens/{mint_address}?chainId=solana"
        try:
            resp = requests.get(url, timeout=10)
            # Print baseToken name if available
            try:
                data = resp.json()
                if data and isinstance(data, dict) and data.get('pairs') and isinstance(data['pairs'], list) and len(data['pairs']) > 0:
                    base_token_name = data['pairs'][0].get('baseToken', {}).get('name', 'N/A')
                    print(f"    baseToken: {base_token_name}")
                else:
                    print("    baseToken: N/A")
            except Exception:
                print("    baseToken: N/A (invalid JSON)")
            # Save the raw JSON string, not the parsed object
            results.append({
                "mint_address": mint_address,
                "usd_volume": row['usd_volume'],
                "dexscreener_data": resp.text
            })
        except Exception as e:
            results.append({
                "mint_address": mint_address,
                "usd_volume": row['usd_volume'],
                "dexscreener_data": f"Error: {e}"
            })
        req_counter += 1
        # Rate limit logic
        if req_counter >= RATE_LIMIT:
            elapsed_window = time.time() - window_start
            if elapsed_window < RATE_PERIOD:
                sleep_time = RATE_PERIOD - elapsed_window
                print(f"Rate limit reached ({RATE_LIMIT} requests). Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            req_counter = 0
            window_start = time.time()
        if idx % 10 == 0:
            print(f"Processed {idx} tokens so far...")
        time.sleep(0.2)  # Be nice to the API
        if TEST_MODE and idx >= TEST_LIMIT:
            break

elapsed = time.time() - start_time
if TEST_MODE:
    print(f"Test mode: Queried {TEST_LIMIT} tokens in {elapsed:.2f} seconds.")

with open(output_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"Done! Results saved to {output_json}") 