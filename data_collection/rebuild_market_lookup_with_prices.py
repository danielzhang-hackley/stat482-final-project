import pandas as pd
import ast
import json

CSV_PATH = "data/markets_data.csv"
OUT_JSON = "data/market_lookup_with_prices.json"

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded markets_data.csv with {len(df)} rows")

    lookup = {}

    for _, row in df.iterrows():
        cond_id = row["condition_id"]

        # New fields you wanted included:
        question = row.get("question")
        question_id = row.get("question_id")

        desc = row.get("description") or question
        market_slug = row.get("market_slug")
        end_date_iso = row.get("end_date_iso")
        accepting_ts = row.get("accepting_order_timestamp")

        tokens_str = row.get("tokens")
        if pd.isna(tokens_str):
            continue

        try:
            tokens_list = ast.literal_eval(tokens_str)
        except Exception as e:
            print(f"Skipping condition_id {cond_id}: failed to parse tokens: {e}")
            continue

        tokens_info = []
        for t in tokens_list:
            tokens_info.append({
                "token_id": t.get("token_id"),
                "outcome": t.get("outcome"),
                "price": t.get("price"),
                "winner": t.get("winner"),
            })

        lookup[cond_id] = {
            "question": question,
            "question_id": question_id,
            "description": desc,
            "market_slug": market_slug,
            "end_date_iso": end_date_iso,
            "accepting_order_timestamp": accepting_ts,
            "tokens": tokens_info,
        }

    with open(OUT_JSON, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"Wrote lookup with prices to {OUT_JSON}")
    print(f"Total markets in lookup: {len(lookup)}")


if __name__ == "__main__":
    main()
