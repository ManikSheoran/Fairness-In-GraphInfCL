# -*- coding: utf-8 -*-
"""
Modified on June 8, 2025
Fairness-aware version of influencer seed selection script
Changes:
- Enforced max influencers per topic during greedy selection
- Added fairness metric (entropy + topic distribution)
- Preserved original logic and added new constraints cleanly
"""

import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import networkx as nx
import argparse
import math

# ------------------- FAIRNESS METRICS -------------------
def compute_entropy(distribution):
    total = sum(distribution.values())
    entropy = -sum((count / total) * math.log2(count / total)
                   for count in distribution.values() if count > 0)
    return entropy

# ------------------- FAIR SEED SELECTION -------------------
def select_fair_seed_set(node_embeddings, second_order_neighbors, topic_map, k, beta, max_per_topic):
    seed_set = set()
    no_embeddings = set()
    node_influence = defaultdict(set)
    topic_count = defaultdict(int)

    for node, neighbors in second_order_neighbors.items():
        if node not in node_embeddings:
            no_embeddings.add(node)
            continue
        filtered_neighbors = set()
        for neighbor in neighbors:
            if neighbor not in node_embeddings:
                no_embeddings.add(neighbor)
                continue
            similarity = cosine_similarity(node_embeddings[node].reshape(1, -1),
                                           node_embeddings[neighbor].reshape(1, -1))[0][0]
            if similarity > beta:
                filtered_neighbors.add(neighbor)
        node_influence[node] = filtered_neighbors

    influenced_nodes = set()
    while len(seed_set) < k:
        best_node = None
        max_influence = 0
        for node, influences in node_influence.items():
            topic = topic_map.get(node, "unknown")
            if topic_count[topic] >= max_per_topic:
                continue
            unique_influences = influences - influenced_nodes
            if len(unique_influences) > max_influence and node not in seed_set:
                best_node = node
                max_influence = len(unique_influences)
        if best_node is None:
            break
        seed_set.add(best_node)
        topic_count[topic_map.get(best_node, "unknown")] += 1
        influenced_nodes.update(node_influence[best_node])
        for node in node_influence:
            node_influence[node] -= influenced_nodes

    return list(seed_set), topic_count

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
    parser.add_argument("--max_per_topic", type=int, default=3)
    parser.add_argument("--topic_map", type=str, default="node_to_topic.pkl")
    args = parser.parse_args()

    with open('node_features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('original_edge_index.pkl', 'rb') as f:
        edge_index = pickle.load(f)
    with open(args.topic_map, 'rb') as f:
        topic_map = pickle.load(f)  # should map node_id to topic string

    second_order_neighbors = get_second_order_neighbors(edge_index)

    seed_nodes, topic_distribution = select_fair_seed_set(
        node_embeddings=features,
        second_order_neighbors=second_order_neighbors,
        topic_map=topic_map,
        k=args.k,
        beta=args.beta,
        max_per_topic=args.max_per_topic
    )

    print("\n Selected Fair Seed Nodes:", seed_nodes)
    print("\n Topic Distribution Among Seeds:")
    for topic, count in topic_distribution.items():
        print(f"  - {topic}: {count}")

    entropy = compute_entropy(topic_distribution)
    print(f"\n Topic Entropy (Fairness Score): {entropy:.4f}")
