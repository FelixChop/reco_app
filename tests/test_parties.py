from scraping.data_collector import fetch_current_french_parties

parties = fetch_current_french_parties()
print("Parties found:")
for pid, label in parties:
    print(f"- {label} ({pid})")
