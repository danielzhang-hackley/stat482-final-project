import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from collections import defaultdict
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import DefaultDict, Dict, List, Set, Tuple, TypeVar


class NomicHybridNLIPipeline:
    def __init__(
        self,
        embed_batch_size=256,
        k_neighbors=30,
        min_cluster_size=10,
        min_samples=5,
        nli_k_neighbors=3,
        nli_batch_size=32,
        entail_threshold=0.70,
        contradiction_threshold=0.45,
        fp16=True,
        device=None,
    ):
        """
        Full Nomic → FAISS → HDBSCAN → Hybrid NLI pipeline.

        - embed_batch_size: Nomic batch size
        - k_neighbors: FAISS neighbors for clustering
        - min_cluster_size, min_samples: HDBSCAN
        - nli_k_neighbors: FAISS neighbors for hybrid NLI refinement
        - nli_batch_size: NLI batch size
        - entail_threshold: similarity cutoff for entailment candidate
        - contradiction_threshold: similarity cutoff for contradiction candidate
        - fp16: run NLI cross-encoder in half precision
        """

        self.embed_batch_size = embed_batch_size
        self.k_neighbors = k_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        self.nli_k_neighbors = nli_k_neighbors
        self.nli_batch_size = nli_batch_size
        self.entail_threshold = entail_threshold
        self.contradiction_threshold = contradiction_threshold

        self.fp16 = fp16
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load embedding model once
        self.embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

    # ----------------------------------------------------------------------
    # Stage 1 — Embeddings
    # ----------------------------------------------------------------------
    def embed(self, df, text_col="statement"):
        print("🔹 Embedding with Nomic...")
        return embed_with_nomic(
            df,
            self.embedding_model,
            text_col=text_col,
            batch_size=self.embed_batch_size,
        )

    # ----------------------------------------------------------------------
    # Stage 2 — Clustering
    # ----------------------------------------------------------------------
    def cluster(self, embeddings):
        print("🔹 Clustering with FAISS + HDBSCAN...")
        labels = cluster_with_nomic(
            embeddings,
            k=self.k_neighbors,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
        )
        return labels

    # ----------------------------------------------------------------------
    # Stage 3 — Hybrid NLI (Dual Thresholds)
    # ----------------------------------------------------------------------
    def build_nli_graph(
        self,
        df,
        embeddings,
        nli_model,
        nli_tokenizer,
    ):
        print("🔹 Running Hybrid Cross-Encoder NLI (dual thresholds)...")

        implied_by, contradictions = hybrid_nli_cosine_dual_thresholds(
            df=df,
            embeddings=embeddings,
            nli_tokenizer=nli_tokenizer,
            nli_model=nli_model,
            k_neighbors=self.nli_k_neighbors,
            batch_size=self.nli_batch_size,
            fp16=self.fp16,
            device=self.device,
            entail_threshold=self.entail_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )
        return implied_by, contradictions

    # ----------------------------------------------------------------------
    # Master Function — Full Pipeline
    # ----------------------------------------------------------------------
    def run(
        self,
        df,
        nli_model,
        nli_tokenizer,
        text_col="statement",
    ):
        """
        Full pipeline execution:
            1. Embed with Nomic
            2. Cluster with FAISS + HDBSCAN
            3. Hybrid NLI inference inside clusters (dual thresholds)

        Returns:
            cluster_labels, implied_by, contradictions
        """
        # Step 1 — Embeddings
        embeddings = self.embed(df, text_col=text_col)

        # Step 2 — Clustering
        cluster_labels = self.cluster(embeddings)
        df["cluster_id"] = cluster_labels

        # Step 3 — Hybrid NLI refinement
        implied_by, contradictions = self.build_nli_graph(
            df, embeddings, nli_model, nli_tokenizer
        )

        return cluster_labels, implied_by, contradictions
    

################################################
# HELPER FUNCTIONS FOR NOMIC PIPELINE
################################################

def embed_with_nomic(df, nomic_model, text_col="statement", batch_size=256):
    embeddings = nomic_model.encode(
        df[text_col].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def faiss_knn_cosine_to_l2(x_norm, k=30):
    """
    x_norm must be L2-normalized.
    Returns L2 distances on unit sphere + nearest neighbor indices.
    """
    x_norm = np.ascontiguousarray(x_norm.astype("float32"))
    N, d = x_norm.shape

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(d))
    else:
        index = faiss.IndexFlatIP(d)

    index.add(x_norm)
    sims, neighbors = index.search(x_norm, k)

    sims = np.clip(sims, -1, 1)
    l2_sq = 2 - 2 * sims
    l2_sq = np.maximum(l2_sq, 0)
    distances = np.sqrt(l2_sq).astype("float32")

    return distances, neighbors


def build_mrd_graph(distances, neighbors):
    N, k = distances.shape
    core_dist = distances[:, -1]

    rows, cols, vals = [], [], []

    for i in range(N):
        for nbr, d in zip(neighbors[i], distances[i]):
            mrd = max(core_dist[i], core_dist[nbr], d)
            if mrd == 0:
                mrd = 1e-9
            rows.append(i); cols.append(nbr); vals.append(mrd)
            rows.append(nbr); cols.append(i); vals.append(mrd)

    return coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()


def hdbscan_components(
    MRD,
    min_cluster_size=10,
    min_samples=5,
    cluster_selection_method="eom"
):
    n_components, comp_labels = connected_components(MRD, directed=False)

    N = MRD.shape[0]
    global_labels = -np.ones(N, dtype=int)
    next_cluster = 0

    for cid in range(n_components):
        idx = np.where(comp_labels == cid)[0]
        if len(idx) == 0:
            continue

        sub_MRD = MRD[idx][:, idx]

        clusterer = HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
        )

        sub_labels = clusterer.fit_predict(sub_MRD)

        for lid in np.unique(sub_labels):
            if lid == -1:
                continue
            mask = sub_labels == lid
            global_labels[idx[mask]] = next_cluster
            next_cluster += 1

    return global_labels

def cluster_with_nomic(
    embeddings,
    k=30,
    min_cluster_size=10,
    min_samples=5,
):
    """
    High-quality clustering pipeline using Nomic embeddings + FAISS cosine + HDBSCAN.
    """

    # embeddings are already normalized by Nomic
    X = embeddings.astype("float32")
    # renormalize to be safe
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

    print("Building FAISS cosine kNN...")
    distances, neighbors = faiss_knn_cosine_to_l2(X, k=k)

    print("Constructing MRD graph...")
    MRD = build_mrd_graph(distances, neighbors)

    print("Running HDBSCAN on components...")
    labels = hdbscan_components(
        MRD,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    return labels


def run_ce_nli_batch(
    pairs, prems, hyps,
    tokenizer, model, device, fp16,
    implied_by, contradictions
):
    if not pairs:
        return

    enc = tokenizer(
        prems, hyps,
        padding=True, truncation=True,
        max_length=256, return_tensors="pt"
    )
    ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device)

    with torch.no_grad():
        if fp16:
            with torch.cuda.amp.autocast():
                logits = model(ids, attention_mask=att).logits
        else:
            logits = model(ids, attention_mask=att).logits

    preds = logits.argmax(dim=1).cpu().numpy()

    for (i, j), label in zip(pairs, preds):
        if label == 0:
            implied_by[j].add(i)
        elif label == 2:
            contradictions[j].add(i)


def hybrid_nli_cosine_dual_thresholds(
    df,
    embeddings,
    nli_tokenizer,
    nli_model,
    k_neighbors=3,
    batch_size=32,
    fp16=True,
    device=None,
    entail_threshold=0.70,        # High similarity required for entailment candidate
    contradiction_threshold=0.45, # Lower similarity allowed for contradictions
):
    """
    Build NLI entailment + contradiction graph using:
        • Nomic embeddings for semantic neighbors
        • FAISS cosine-based kNN for candidate pairs
        • Dual-threshold filtering:
            - High threshold for entailment candidates
            - Lower threshold for contradiction candidates

    Returns:
        implied_by: dict(j) -> set(i) meaning: i entails j
        contradictions: dict(j) -> set(i) meaning: i contradicts j
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nli_model = nli_model.to(device).eval()
    if fp16:
        nli_model.half()

    texts = df["statement"].tolist()
    implied_by = defaultdict(set)
    contradictions = defaultdict(set)

    # Clusters: df["cluster_id"] must already be assigned
    clusters = df.groupby("cluster_id").indices

    X = embeddings.astype("float32")  # Already normalized if using Nomic

    for cid, idxs in tqdm(clusters.items(), desc="Hybrid NLI (dual thresholds)"):
        idxs = np.array(idxs)
        if len(idxs) < 2:
            continue

        # ----------------------------------------------------------------------
        # Step 1: FAISS cosine kNN
        # ----------------------------------------------------------------------
        cluster_emb = X[idxs]
        dists, nbrs = faiss_knn_cosine_to_l2(cluster_emb, k=min(k_neighbors, len(idxs)))

        # Convert L2 on the unit sphere back to cosine similarity:
        sims = 1 - (dists**2) / 2

        # Candidate buffers
        entail_pairs, entail_prems, entail_hyps = [], [], []
        contra_pairs, contra_prems, contra_hyps = [], [], []

        # ----------------------------------------------------------------------
        # Step 2: Dual-threshold pair selection
        # ----------------------------------------------------------------------
        for i_local, i_global in enumerate(idxs):
            prem_text = texts[i_global]

            for rank in range(nbrs.shape[1]):
                j_local = nbrs[i_local, rank]
                if j_local == i_local:
                    continue

                sim = sims[i_local, rank]
                j_global = idxs[j_local]
                hyp_text = texts[j_global]

                # ENTailment candidate
                if sim >= entail_threshold:
                    entail_pairs.append((i_global, j_global))
                    entail_prems.append(prem_text)
                    entail_hyps.append(hyp_text)

                # CONTRAdiCtion candidate
                elif sim >= contradiction_threshold:
                    contra_pairs.append((i_global, j_global))
                    contra_prems.append(prem_text)
                    contra_hyps.append(hyp_text)

                # Else: similarity too low → skip

        # ----------------------------------------------------------------------
        # Step 3: Run NLI on entailment candidates
        # ----------------------------------------------------------------------
        for start in range(0, len(entail_pairs), batch_size):
            batch_pairs = entail_pairs[start:start+batch_size]
            batch_prem = entail_prems[start:start+batch_size]
            batch_hyp = entail_hyps[start:start+batch_size]

            if not batch_pairs:
                continue

            enc = nli_tokenizer(
                batch_prem, batch_hyp,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                if fp16:
                    with torch.cuda.amp.autocast():
                        logits = nli_model(**enc).logits
                else:
                    logits = nli_model(**enc).logits

            preds = logits.argmax(dim=1).cpu().numpy()

            for (i, j), label in zip(batch_pairs, preds):
                if label == 0:  # entailment
                    implied_by[j].add(i)

        # ----------------------------------------------------------------------
        # Step 4: Run NLI on contradiction candidates
        # ----------------------------------------------------------------------
        for start in range(0, len(contra_pairs), batch_size):
            batch_pairs = contra_pairs[start:start+batch_size]
            batch_prem = contra_prems[start:start+batch_size]
            batch_hyp = contra_hyps[start:start+batch_size]

            if not batch_pairs:
                continue

            enc = nli_tokenizer(
                batch_prem, batch_hyp,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                if fp16:
                    with torch.cuda.amp.autocast():
                        logits = nli_model(**enc).logits
                else:
                    logits = nli_model(**enc).logits

            preds = logits.argmax(dim=1).cpu().numpy()

            for (i, j), label in zip(batch_pairs, preds):
                if label == 2:  # contradiction
                    contradictions[j].add(i)

    return implied_by, contradictions