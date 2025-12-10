import os
import time
import logging
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
import json




# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load environment variables from the .env file
load_dotenv("keys.env")

# Access the environment variables
API_KEY = os.getenv('API_KEY')
LOOKUP_PATH = "data/market_lookup.json" 

try:
    with open(LOOKUP_PATH, "r") as f:
        MARKET_LOOKUP = json.load(f)
    print(f"Loaded {len(MARKET_LOOKUP)} markets from lookup.")
except Exception as e:
    print(f"Could not load lookup JSON: {e}")
    MARKET_LOOKUP = {}

host = "https://clob.polymarket.com"
chain_id = 137  # Polygon Mainnet

print(f"Using API key: {API_KEY}")

# Initialize the ClobClient
client = ClobClient(host, chain_id=chain_id)

# Dictionary to cache live prices
live_price_cache = {}
CACHE_DURATION = 60  # Cache live prices for 1 minute


def get_live_price(token_id):
    """
    Fetch the live price for a given token ID.

    Args:
        token_id (str): The token ID for which the live price is being requested.

    Returns:
        float: The live price for the given token ID.
    """
    cache_key = f"{token_id}"
    current_time = time.time()

    # Check if the price is in the cache and still valid
    if cache_key in live_price_cache:
        cached_price, timestamp = live_price_cache[cache_key]
        if current_time - timestamp < CACHE_DURATION:
            logger.info(f"Returning cached price for {cache_key}: {cached_price}")
            return cached_price
        else:
            logger.info(f"Cache expired for {cache_key}. Fetching a new price.")

    # Fetch new price from the API
    try:
        response = client.get_last_trade_price(token_id=token_id)
        price = response.get('price')

        # Cache the price with the current timestamp
        live_price_cache[cache_key] = (price, current_time)
        logger.info(f"Fetched live price for {cache_key}: {price}")
        return price
    except Exception as e:
        logger.error(f"Failed to fetch live price for token {token_id}: {str(e)}")
        return None

def list_markets():
    """Return (cond_id, description) for all markets."""
    return [(cid, info["description"]) for cid, info in MARKET_LOOKUP.items()]

def get_first_token_for_condition_id(cond_id):
    """Return first token_id for a given condition_id."""
    info = MARKET_LOOKUP.get(cond_id)
    if not info:
        return None

    tokens = info.get("tokens")
    if not tokens or len(tokens) == 0:
        return None

    return tokens[0]["token_id"]

# If this script is executed directly, it can take command-line arguments to test the live price retrieval
if __name__ == "__main__":
    import sys

    # If token_id passed → use it
    if len(sys.argv) >= 2:
        token_id = sys.argv[1]
    else:
        # No token passed → show menu
        print("\nAvailable markets:")
        markets = list_markets()

        # Show first 20 markets only so it's not overwhelming
        for i, (cid, desc) in enumerate(markets[:20]):
            print(f"{i}: {desc} ({cid[:10]}...)")

        choice = int(input("\nSelect market index: "))
        cond_id = markets[choice][0]
        token_id = get_first_token_for_condition_id(cond_id)

        print(f"Selected market: {markets[choice][1]}")
        print(f"Using token_id: {token_id}")

    live_price = get_live_price(token_id)
    if live_price is not None:
        print(f"Live price for token {token_id}: {live_price}")
    else:
        print(f"Could not fetch the live price for token {token_id}.")
