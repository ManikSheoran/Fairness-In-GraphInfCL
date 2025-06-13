# -*- coding: utf-8 -*-
"""
Modified on June 8, 2025
Fairness-aware influencer seed selection script
Updates:
- Removed topic cap constraint
- Selects top-k influencers based on influence size and gender fairness
- Includes gender balance metric per influencer
"""

import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import networkx as nx
import argparse
import math

# ------------------- FAIRNESS METRICS -------------------
def compute_gender_balance(influenced_nodes, gender_map):
    gender_counts = Counter([gender_map.get(node, 'unknown') for node in influenced_nodes])
    male = gender_counts.get('male', 0)
    female = gender_counts.get('female', 0)
    total = male + female
    if total == 0:
        return 0.0
    return 1.0 - abs((male / total) - (female / total))

# ------------------- SEED SELECTION -------------------
def select_top_fair_seeds(node_embeddings, second_order_neighbors, gender_map, k, beta):
    node_influence = defaultdict(set)
    node_scores = []

    for node, neighbors in second_order_neighbors.items():
        if node not in node_embeddings:
            continue
        filtered_neighbors = set()
        for neighbor in neighbors:
            if neighbor not in node_embeddings:
                continue
            similarity = cosine_similarity(node_embeddings[node].reshape(1, -1),
                                           node_embeddings[neighbor].reshape(1, -1))[0][0]
            if similarity > beta:
                filtered_neighbors.add(neighbor)
        if filtered_neighbors:
            balance_score = compute_gender_balance(filtered_neighbors, gender_map)
            total_influence = len(filtered_neighbors)
            node_scores.append((node, total_influence, balance_score))

    # Sort by influence first, then by gender balance
    sorted_nodes = sorted(node_scores, key=lambda x: (x[1], x[2]), reverse=True)
    top_k = sorted_nodes[:k]
    seed_nodes = [entry[0] for entry in top_k]
    gender_balances = {entry[0]: entry[2] for entry in top_k}

    return seed_nodes, gender_balances

# ------------------- SUPPORTING UTILITIES -------------------
def get_second_order_neighbors(edge_index):
    G = nx.DiGraph()
    G.add_edges_from(edge_index)
    second_order_neighbors = defaultdict(set)
    for node in G.nodes():
        first_order = set(G.successors(node))
        second_order = set()
        for neighbor in first_order:
            second_order.update(G.successors(neighbor))
        second_order.discard(node)
        second_order_neighbors[node] = second_order
    return second_order_neighbors

# ------------------- MAIN EXECUTION -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--gender_map", type=str, default="node_to_gender.pkl")
    args = parser.parse_args()

    with open('node_features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('original_edge_index.pkl', 'rb') as f:
        edge_index = pickle.load(f)
    with open(args.gender_map, 'rb') as f:
        gender_map = pickle.load(f)

    second_order_neighbors = get_second_order_neighbors(edge_index)

    seed_nodes, gender_balances = select_top_fair_seeds(
        node_embeddings=features,
        second_order_neighbors=second_order_neighbors,
        gender_map=gender_map,
        k=args.k,
        beta=args.beta
    )

    print("\nSelected Influencers (Top-k Influence + Gender Balance):")
    for node in seed_nodes:
        print(f"  - Node {node} → Gender Balance Score: {gender_balances[node]:.2f}")
