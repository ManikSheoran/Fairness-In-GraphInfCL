import pickle
import numpy as np
from sklearn.cluster import KMeans

# Load node embeddings
with open("node_features.pkl", "rb") as f:
    feature_dict = pickle.load(f)

# Prepare data
node_ids = list(feature_dict.keys())
embeddings = np.array([feature_dict[nid] for nid in node_ids])

# Cluster into topics
num_topics = 5
kmeans = KMeans(n_clusters=num_topics, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Create node → topic map
node_to_topic = {node_id: f"topic_{label}" for node_id, label in zip(node_ids, labels)}

# Save as pickle
with open("node_to_topic.pkl", "wb") as f:
    pickle.dump(node_to_topic, f)

print("node_to_topic.pkl has been generated.")
