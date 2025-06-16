import json
import glob
import csv
import os

# Directory containing the JSON files
data_dir = os.path.dirname(__file__)
# Output CSV file
output_csv = os.path.join(data_dir, "solana_mint_volume.csv")

# Find all relevant JSON files
json_files = sorted(glob.glob(os.path.join(data_dir, "results_page*.json")))

rows = []

for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        # The structure is: data['result']['rows'] is a list of dicts
        for row in data['result']['rows']:
            rows.append({
                "mint_address": row["mint_address"],
                "usd_volume": row["usd_volume"]
            })

# Write to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["mint_address", "usd_volume"])
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV written to {output_csv}") 