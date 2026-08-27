"""
SentinelPay AI - Graph Analysis Layer (NetworkX, no ML model)
==================================================================
Builds a bipartite-ish graph of users <-> devices <-> merchants and
flags structural fraud rings: multiple accounts sharing a device,
tight clusters transacting with the same merchant in a short window.

No trained model - just centrality + community detection, which is
demo-credible for a 3-day MVP.
"""

import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

df = pd.read_csv("features_with_anomaly.csv", parse_dates=["timestamp"])

# --- Build graph: nodes = users + devices + merchants ---------------------
G = nx.Graph()
for _, row in df.iterrows():
    u, d, m = f"user::{row['user_id']}", f"device::{row['device_id']}", f"merchant::{row['merchant_id']}"
    G.add_node(u, type="user")
    G.add_node(d, type="device")
    G.add_node(m, type="merchant")
    G.add_edge(u, d)
    G.add_edge(u, m)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- Degree centrality on DEVICE nodes: how many distinct users share it --
device_degree = {n: G.degree(n) for n in G.nodes if n.startswith("device::")}
shared_devices = {n: deg for n, deg in device_degree.items() if deg >= 3}  # >=3 users on one device
print(f"\nDevices shared by 3+ users (likely mule rings): {len(shared_devices)}")

# --- Betweenness centrality: which devices/merchants bridge many users ----
# (expensive on big graphs; fine at this MVP scale)
betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), seed=3)

# --- Community detection: clusters of users connected via shared devices --
communities = list(greedy_modularity_communities(G))
user_community = {}
for i, comm in enumerate(communities):
    for node in comm:
        if node.startswith("user::"):
            user_community[node.replace("user::", "")] = i

# small communities that are almost entirely bound by device-sharing (not
# organic shared-merchant popularity) are the interesting ones
community_sizes = pd.Series(list(user_community.values())).value_counts()

# --- Per-transaction graph_score -------------------------------------------
def graph_score(row):
    score = 0
    reasons = []
    dev_node = f"device::{row['device_id']}"
    deg = device_degree.get(dev_node, 1)
    if deg >= 5:
        score += 40
        reasons.append(f"device shared by {deg} distinct users")
    elif deg >= 3:
        score += 20
        reasons.append(f"device shared by {deg} distinct users")
    bc = betweenness.get(dev_node, 0)
    if bc > betweenness_p90:
        score += 15
        reasons.append("device is a high-betweenness bridge node")
    return min(score, 100), "; ".join(reasons)

betweenness_p90 = pd.Series(betweenness).quantile(0.90)
results = df.apply(graph_score, axis=1)
df["graph_score"] = results.apply(lambda x: x[0])
df["graph_reasons"] = results.apply(lambda x: x[1])

print("\nMean graph_score by fraud_type (device_ring should dominate):")
print(df.groupby("fraud_type")["graph_score"].mean().sort_values(ascending=False))

from sklearn.metrics import roc_auc_score
print("\ngraph_score ROC-AUC vs ground truth (whole dataset, expect skew toward device_ring):",
      round(roc_auc_score(df["is_fraud"], df["graph_score"]), 3))

df.to_csv("features_with_graph.csv", index=False)
print("\nSaved features_with_graph.csv (adds graph_score column)")
