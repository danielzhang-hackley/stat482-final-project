#%%
#Setup
#pip install json
#pip install time
#pip install typing
#pip install requests

#%%
import json
import time
from typing import Any, Dict, List, Optional
import requests

base_url = "https://data-api.polymarket.com/trades"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
#%%


#%%
# Sum Yes < 1 check helpers
# ------------------------------------------------------------
# 1. Fetch events from Gamma
# ------------------------------------------------------------

def fetch_events(
    limit: int = 100,
    max_pages: int = 5,
    closed: bool = False,
    sleep_secs: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Fetch a batch of events from the Gamma API.
    """
    events: List[Dict[str, Any]] = []
    offset = 0

    for _ in range(max_pages):
        params = {
            "limit": limit,
            "offset": offset,
            "closed": str(closed).lower(),
        }
        resp = requests.get(f"{GAMMA_BASE_URL}/events", params=params, timeout=10)
        resp.raise_for_status()
        page = resp.json()

        if not page:
            break

        events.extend(page)
        offset += limit
        time.sleep(sleep_secs)

    return events


def filter_negative_risk_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep events that are:
      - negRisk = True
      - active and not closed
      - CLOB-enabled (enableOrderBook = True)
      - have at least 2 markets
    """
    filtered = []
    for ev in events:
        if not ev.get("negRisk"):
            continue
        if not ev.get("active", False):
            continue
        if ev.get("closed", False):
            continue

        markets = ev.get("markets") or []
        if len(markets) < 2:
            continue

        if not ev.get("enableOrderBook", False):
            continue

        filtered.append(ev)

    return filtered


# ------------------------------------------------------------
# 2. Helpers to parse Gamma JSON-string fields and call /price
# ------------------------------------------------------------

def parse_json_field(obj: Dict[str, Any], field: str):
    """
    Some Gamma fields are JSON strings (e.g., outcomes, outcomePrices, clobTokenIds).
    This helper safely parses them into Python lists.
    """
    raw = obj.get(field)
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return []


def get_yes_token_id(market: Dict[str, Any]) -> Optional[str]:
    """
    For a binary market, return the YES token id.

    Convention (checked against UI):
      - outcomes[0] is the YES-like leg (e.g. "Yes" or named candidate)
      - clobTokenIds[0] is its token_id
      - clobTokenIds[1] is the opposite leg (No or complement)

    We require at least two outcomes and two clobTokenIds.
    """
    outcomes = parse_json_field(market, "outcomes")
    clob_ids = parse_json_field(market, "clobTokenIds")

    if len(outcomes) < 2 or len(clob_ids) < 2:
        return None

    # Require binary structure, but don't enforce label == "Yes"
    return clob_ids[0]


def get_best_ask_for_token(token_id: str) -> Optional[float]:
    """
    Get the best executable ask for a given YES token from CLOB.
    For Polymarket's /price API, side=SELL corresponds to the UI "Buy Yes" price.
    """
    try:
        resp = requests.get(
            f"{CLOB_BASE_URL}/price",
            params={"token_id": token_id, "side": "SELL"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data["price"])
    except Exception:
        return None


# ------------------------------------------------------------
# 3. Compute basket YES cost for each negRisk event
# ------------------------------------------------------------

def compute_yes_basket_cost_for_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    For one negRisk event, compute the real basket YES cost:

        basket_cost = sum over markets of best_ask(YES_token)

    Returns a summary dict with:
      - event_id, slug, title
      - n_markets_used
      - basket_yes_cost
    """
    markets = event.get("markets") or []

    basket_cost = 0.0
    n_used = 0

    for m in markets:
        # Optionally skip markets that are not accepting orders
        if not m.get("acceptingOrders", True):
            continue

        yes_token_id = get_yes_token_id(m)
        if yes_token_id is None:
            continue

        ask = get_best_ask_for_token(yes_token_id)
        if ask is None:
            continue

        basket_cost += ask
        n_used += 1

    return {
        "event_id": event.get("id"),
        "slug": event.get("slug"),
        "title": event.get("title"),
        "n_markets_used": n_used,
        "basket_yes_cost": basket_cost,
    }


def compute_yes_basket_costs_for_events(
    limit: int = 100,
    max_pages: int = 5,
    max_events: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    High-level driver:
      - fetch events from Gamma
      - filter to negative-risk, CLOB-enabled events
      - for each event compute basket YES cost

    Returns list of summary dicts, one per event.
    """
    events = fetch_events(limit=limit, max_pages=max_pages, closed=False)
    neg_events = filter_negative_risk_events(events)

    results: List[Dict[str, Any]] = []
    count = 0

    for ev in neg_events:
        if max_events is not None and count >= max_events:
            break

        summary = compute_yes_basket_cost_for_event(ev)
        results.append(summary)
        count += 1

        if count % 5 == 0:
            print(f"Computed basket cost for {count} negRisk events...")

    return results
#%%
#Run Sum Yes < 1 check
def main():
    summaries = compute_yes_basket_costs_for_events(
        limit=100,
        max_pages=5,
        max_events=None,  # or set a small number for testing
    )

    print("\n=== YES basket costs for negRisk events ===")
    for s in sorted(summaries, key=lambda x: x["basket_yes_cost"]):
        print(
            f"basket_yes_cost={s['basket_yes_cost']:.4f} | "
            f"n_markets={s['n_markets_used']:2d} | "
            f"slug={s['slug']} | title={s['title']}"
        )

    print("\nPotential theoretical basket-buy opportunities (basket_yes_cost < 1):")
    for s in sorted(summaries, key=lambda x: x["basket_yes_cost"]):
        if s["basket_yes_cost"] < 1.0 and s["n_markets_used"] >= 2:
            print(
                f"- cost={s['basket_yes_cost']:.4f} | "
                f"n_markets={s['n_markets_used']:2d} | slug={s['slug']}"
            )


if __name__ == "__main__":
    main()
#%%

#Binary market check helper
#You can adjust max_pages to adjust the number of games you want to analyze, each page = 1000 games
def fetch_clob_markets(max_pages: int = 5, sleep_secs: float = 0.1) -> List[Dict[str, Any]]:
    """
    Fetch a list of CLOB markets .
    """
    markets: List[Dict[str, Any]] = []
    cursor: str = ""

    for _ in range(max_pages):
        params = {}
        if cursor:
            params["next_cursor"] = cursor

        resp = requests.get(f"{CLOB_BASE_URL}/markets", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        markets.extend(data.get("data", []))
        cursor = data.get("next_cursor") or ""

        if cursor == "LTE=":
            break

        time.sleep(sleep_secs)

    return markets


def get_best_prices_for_token(token_id: str) -> Optional[Tuple[float, float]]:
    """
    Get best bid and best ask for a given token_id.
    Returns (bid, ask) or None on error.
    """
    try:
        # Best ask (you BUY, so side = BUY)
        r_ask = requests.get(
            f"{CLOB_BASE_URL}/price",
            params={"token_id": token_id, "side": "BUY"},
            timeout=5,
        )
        r_ask.raise_for_status()
        ask_price = float(r_ask.json()["price"])

        # Best bid (you SELL, so side = SELL)
        r_bid = requests.get(
            f"{CLOB_BASE_URL}/price",
            params={"token_id": token_id, "side": "SELL"},
            timeout=5,
        )
        r_bid.raise_for_status()
        bid_price = float(r_bid.json()["price"])

        return bid_price, ask_price
    except Exception as e:
        # You can uncomment this to see details while debugging:
        # print(f"Error fetching prices for token {token_id}: {e}")
        return None


def check_binary_arb(
    best_bid_a: float,
    best_ask_a: float,
    best_bid_b: float,
    best_ask_b: float,
    fee_buffer: float = 0.002,
):
    """
    Simple binary arb check for one YES/NO pair.
    """
    buy_basket_cost = best_ask_a + best_ask_b
    sell_basket_proceeds = best_bid_a + best_bid_b

    buy_side_arb = max(0.0, 1.0 - fee_buffer - buy_basket_cost)
    sell_side_arb = max(0.0, sell_basket_proceeds - 1.0 - fee_buffer)

    return {
        "buy_basket_cost": buy_basket_cost,
        "sell_basket_proceeds": sell_basket_proceeds,
        "buy_side_arb": buy_side_arb,
        "sell_side_arb": sell_side_arb,
    }


def scan_binary_markets_for_arb(
    fee_buffer: float = 0.002,
    min_edge: float = 0.001,
    max_markets: int = 50,
):
    markets = fetch_clob_markets()
    print(f"Fetched {len(markets)} CLOB markets")

    checked = 0
    for i, m in enumerate(markets):
        if checked >= max_markets:
            break

        tokens = m.get("tokens") or []
        if len(tokens) != 2:
            continue  # not a standard binary pair

        token_a_id = tokens[0]["token_id"]
        token_b_id = tokens[1]["token_id"]

        prices_a = get_best_prices_for_token(token_a_id)
        prices_b = get_best_prices_for_token(token_b_id)
        
        checked += 1
        if checked % 10 == 0:
            print(f"Checked {checked} binary markets...")

        if prices_a is None or prices_b is None:
            continue

        bid_a, ask_a = prices_a
        bid_b, ask_b = prices_b

        result = check_binary_arb(bid_a, ask_a, bid_b, ask_b, fee_buffer=fee_buffer)

        if (result["buy_side_arb"] > min_edge) or (result["sell_side_arb"] > min_edge):
            print("\n=== Potential arbitrage market ===")
            print(f"Question: {m.get('question')}")
            print(f"Slug:     {m.get('market_slug')}")
            print(f"Token A:  {tokens[0]['outcome']} | token_id={token_a_id}")
            print(f"  bid={bid_a:.4f}, ask={ask_a:.4f}")
            print(f"Token B:  {tokens[1]['outcome']} | token_id={token_b_id}")
            print(f"  bid={bid_b:.4f}, ask={ask_b:.4f}")
            print(
                f"Buy basket cost:      {result['buy_basket_cost']:.4f} "
                f"(arb={result['buy_side_arb']:.4f})"
            )
            print(
                f"Sell basket proceeds: {result['sell_basket_proceeds']:.4f} "
                f"(arb={result['sell_side_arb']:.4f})"
            )

    print(f"\nDone. Checked {checked} binary markets.")
#%%
#Run Binary Market Check
if __name__ == "__main__":
    scan_binary_markets_for_arb(
        fee_buffer=0.002,
        min_edge=0.001,
        max_markets=500,
    )
#%%


