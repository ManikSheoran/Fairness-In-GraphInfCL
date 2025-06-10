# Fairness in GraphInfCL
### Fairness-Aware Influencer Selection

Added a **topic cap constraint** during seed selection to avoid overrepresentation from any one topic group.

**Changes Made**

- Introduced a `max_per_topic` constraint during seed selection, to cap the number of influencers from one topic.
- Clustered node embeddings into topic groups using **KMeans**.
- Generated `node_to_topic.pkl` to assign each node a topic label.
- Modified `seed_tag.py` to enforce topic caps during selection.
- Added a fairness metric (**topic entropy**) to measure diversity in the selected seed set.

I used the `node_features.pkl` file (which contains node embeddings) and clustered them into **5 topic groups** using KMeans. (No. of topics is manually selected.)

```python
import pickle
import numpy as np
from sklearn.cluster import KMeans

with open("node_features.pkl", "rb") as f:
    feature_dict = pickle.load(f)

node_ids = list(feature_dict.keys())
embeddings = np.array([feature_dict[nid] for nid in node_ids])

# Cluster into topics
num_topics = 5
kmeans = KMeans(n_clusters=num_topics, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Create node → topic map
node_to_topic = {node_id: f"topic_{label}" for node_id, label in zip(node_ids, labels)}

with open("node_to_topic.pkl", "wb") as f:
    pickle.dump(node_to_topic, f)

print("node_to_topic.pkl has been generated.")

```

We got `node_to_topic.pkl` file with embedding. This files is used for selecting influencers based on topic in seed_tag.py.

### Run the Code —

```bash
python seed_tag.py --k 10 --beta 0.1 --max_per_topic 3 --topic_map node_to_topic.pkl
```

- `-k` = number of influencers to select
- `-beta` = cosine similarity threshold
- `-max_per_topic` = maximum influencers allowed per topic group
- `-topic_map` = topic cluster file you generated above

Example output:

```bash
Topic Distribution:
  - topic_0: 0
  - topic_1: 3
  - topic_2: 3
  - topic_3: 1
  - topic_4: 3
Topic Entropy (Fairness Score): 1.8955
```
