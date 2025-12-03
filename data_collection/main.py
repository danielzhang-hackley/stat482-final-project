import requests
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path

# ==================== config ====================
API_URL = "https://clob.polymarket.com/markets"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.80  # Markets above this are considered similar
MIN_CLUSTER_SIZE = 2

# File paths for resumable pipeline
MARKETS_RAW_PATH = OUTPUT_DIR / "markets_raw.json"
MARKETS_PROCESSED_PATH = OUTPUT_DIR / "markets_processed.json"
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"

# ==================== 1. fetch market ====================
def fetch_all_markets(force_refresh=False):
    """Fetch all markets from Polymarket API (or load from cache)"""
    
    if MARKETS_RAW_PATH.exists() and not force_refresh:
        print(f"📁 Loading cached markets from {MARKETS_RAW_PATH}")
        with open(MARKETS_RAW_PATH, 'r') as f:
            markets = json.load(f)
        print(f"✓ Loaded {len(markets)} markets from cache")
        return markets
    
    print("Fetching markets from Polymarket API...")
    try:
        r = requests.get(API_URL, timeout=10)
        r.raise_for_status()
        markets = r.json()
        
        # Handle possible structures
        if isinstance(markets, dict):
            if 'data' in markets:
                markets = markets['data']
            elif 'markets' in markets:
                markets = markets['markets']
        
        print(f"✓ Fetched {len(markets)} markets")
        save_markets_raw(markets)
        return markets
    except Exception as e:
        print(f"✗ Error fetching markets: {e}")
        return []

def save_markets_raw(markets):
    """Save raw market data to JSON"""
    with open(MARKETS_RAW_PATH, 'w') as f:
        json.dump(markets, f, indent=2)
    print(f"✓ Saved raw markets to {MARKETS_RAW_PATH}")

# ==================== 2. PREPROCESS ====================
def preprocess_text(text):
    """Clean and normalize market text"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def extract_market_info(markets, force_refresh=False):
    """Extract relevant fields and preprocess text (or load from cache)"""
    
    if MARKETS_PROCESSED_PATH.exists() and not force_refresh:
        print(f"📁 Loading cached processed markets from {MARKETS_PROCESSED_PATH}")
        with open(MARKETS_PROCESSED_PATH, 'r') as f:
            processed = json.load(f)
        print(f"✓ Loaded {len(processed)} processed markets from cache")
        return processed
    
    print("Preprocessing markets...")
    processed = []
    
    for m in markets:
        if not isinstance(m, dict):
            print(f"⚠ Skipping non-dict item: {type(m)}")
            continue
        
        text = m.get('question', '')
        if not text:
            continue

        tokens = []
        for t in m.get('tokens', []):
            tokens.append({
                'token_id': t.get('token_id', ''),
                'outcome': t.get('outcome', ''),
                'price': t.get('price', 0),
                'winner': t.get('winner', False)
            })

        processed.append({
            'market_id': m.get('condition_id', ''),
            'market_slug': m.get('market_slug', ''),
            'original_text': text,
            'processed_text': preprocess_text(text),
            'active': m.get('active', False),
            'closed': m.get('closed', False),
            'tags': m.get('tags', []),
            'tokens': tokens
        })
    
    print(f"✓ Preprocessed {len(processed)} markets")
    
    with open(MARKETS_PROCESSED_PATH, 'w') as f:
        json.dump(processed, f, indent=2)
    print(f"✓ Saved processed markets to {MARKETS_PROCESSED_PATH}")
    
    return processed

# ==================== 3. GENERATE EMBEDDINGS ====================
def generate_embeddings(processed_markets, force_refresh=False):
    
    if EMBEDDINGS_PATH.exists() and not force_refresh:
        print(f"📁 Loading cached embeddings from {EMBEDDINGS_PATH}")
        embeddings = np.load(EMBEDDINGS_PATH)
        print(f"✓ Loaded embeddings: {embeddings.shape}")
        return embeddings
    
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [m['processed_text'] for m in processed_markets]
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"✓ Generated and saved embeddings: {embeddings.shape}")
    
    return embeddings

# ==================== 4. CLUSTERING ====================
def cluster_markets(embeddings, processed_markets):
    """Cluster markets using agglomerative clustering"""
    print("\nClustering markets...")
    
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - similarity_matrix
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - SIMILARITY_THRESHOLD,
        metric='precomputed',
        linkage='average'
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    for i, market in enumerate(processed_markets):
        market['cluster_id'] = int(labels[i])
    
    n_clusters = len(set(labels))
    print(f"✓ Found {n_clusters} clusters")
    
    return labels, similarity_matrix

def find_similar_pairs(embeddings, processed_markets, similarity_matrix):
    """Find high-similarity pairs for manual review"""
    pairs = []
    n = len(processed_markets)
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i][j]
            if sim >= SIMILARITY_THRESHOLD:
                pairs.append({
                    'market_1_id': processed_markets[i]['market_id'],
                    'market_1_text': processed_markets[i]['original_text'],
                    'market_2_id': processed_markets[j]['market_id'],
                    'market_2_text': processed_markets[j]['original_text'],
                    'similarity': round(sim, 4),
                    'same_cluster': processed_markets[i]['cluster_id'] == processed_markets[j]['cluster_id']
                })
    
    pairs.sort(key=lambda x: x['similarity'], reverse=True)
    print(f"✓ Found {len(pairs)} similar pairs (similarity >= {SIMILARITY_THRESHOLD})")
    
    return pairs

# ==================== 5. EXPORT ====================
def export_clusters(processed_markets):
    df = pd.DataFrame(processed_markets)
    df = df.sort_values(['cluster_id', 'processed_text'])
    
    output_path = OUTPUT_DIR / "clusters_for_validation.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved clusters to {output_path}")
    
    cluster_summary = df.groupby('cluster_id').agg({
        'market_id': 'count',
        'original_text': lambda x: ' | '.join(x[:3])
    }).rename(columns={'market_id': 'market_count', 'original_text': 'examples'})
    
    summary_path = OUTPUT_DIR / "cluster_summary.csv"
    cluster_summary.to_csv(summary_path)
    print(f"✓ Saved cluster summary to {summary_path}")
    
    return df

def export_similar_pairs(pairs):
    if not pairs:
        print("ℹ No similar pairs to export")
        return None
    df = pd.DataFrame(pairs)
    output_path = OUTPUT_DIR / "similar_pairs.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved similar pairs to {output_path}")
    return df

def create_validation_template(processed_markets):
    df = pd.DataFrame(processed_markets)
    df['equivalence_group'] = ''
    df['notes'] = ''
    
    template = df[['market_id', 'original_text', 'cluster_id', 'tags', 
                   'active', 'equivalence_group', 'notes']]
    
    output_path = OUTPUT_DIR / "manual_validation_template.csv"
    template.to_csv(output_path, index=False)
    print(f"✓ Created validation template at {output_path}")
    print("\n📋 NEXT STEP: Edit 'manual_validation_template.csv'")
    print("   - Fill in 'equivalence_group' column with group names")
    print("   - Use same name for markets that are TRUE equivalents")
    print("   - Leave blank for markets that don't have equivalents")
    
    return template

# ==================== DEBUG ====================
def inspect_raw_data():
    print("\n" + "=" * 60)
    print("DEBUG: INSPECTING RAW MARKETS DATA")
    print("=" * 60)
    
    if not MARKETS_RAW_PATH.exists():
        print("✗ No raw markets file found. Run pipeline first.")
        return
    
    with open(MARKETS_RAW_PATH, 'r') as f:
        data = json.load(f)
    
    print(f"Type of data: {type(data)}")
    print(f"Length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
    
    if isinstance(data, list) and len(data) > 0:
        print(f"\nFirst item type: {type(data[0])}")
        print(f"First item sample:")
        print(json.dumps(data[0], indent=2)[:500])
    elif isinstance(data, dict):
        print(f"\nDict keys: {list(data.keys())}")
        for key in list(data.keys())[:3]:
            print(f"\n{key}: {type(data[key])}")
    
    print("=" * 60)

# ==================== MAIN PIPELINE ====================
def main(force_refresh=False):
    print("=" * 60)
    print("POLYMARKET MISPRICING DETECTION - CLUSTERING PIPELINE")
    print("=" * 60)
    
    markets = fetch_all_markets(force_refresh=force_refresh)
    if not markets:
        print("✗ No markets available. Exiting.")
        return
    
    if len(markets) > 0 and not isinstance(markets[0], dict):
        print("\n⚠ Data format issue detected. Running inspector...")
        inspect_raw_data()
        return
    
    processed_markets = extract_market_info(markets, force_refresh=force_refresh)
    if not processed_markets:
        print("✗ No processed markets. Exiting.")
        return
    
    embeddings = generate_embeddings(processed_markets, force_refresh=force_refresh)
    
    labels, similarity_matrix = cluster_markets(embeddings, processed_markets)
    
    pairs = find_similar_pairs(embeddings, processed_markets, similarity_matrix)
    
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)
    
    export_clusters(processed_markets)
    export_similar_pairs(pairs)
    create_validation_template(processed_markets)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total markets: {len(processed_markets)}")
    print(f"Clusters found: {len(set(labels))}")
    print(f"Similar pairs (>{SIMILARITY_THRESHOLD}): {len(pairs)}")
    
    cluster_sizes = pd.Series([m['cluster_id'] for m in processed_markets]).value_counts().head(10)
    print("\nTop 10 Largest Clusters:")
    for cluster_id, size in cluster_sizes.items():
        if size >= MIN_CLUSTER_SIZE:
            examples = [m['original_text'][:60] for m in processed_markets if m['cluster_id'] == cluster_id][:2]
            print(f"  Cluster {cluster_id}: {size} markets")
            for ex in examples:
                print(f"    - {ex}...")
    
    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR.absolute()}")
    print("\nNext: Review and edit 'manual_validation_template.csv'")

if __name__ == "__main__":
    import sys
    force = '--force' in sys.argv or '-f' in sys.argv
    debug = '--debug' in sys.argv or '-d' in sys.argv
    
    if debug:
        inspect_raw_data()
    else:
        main(force_refresh=force)
