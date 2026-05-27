import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
from collections import defaultdict, deque

def compute_semantic_MST(texts, model=None, distance_matrix=None):
    if distance_matrix is None:
        r = len(texts)
        m_distance = model.similarity(model.encode(texts), model.encode(texts))
        torch.cuda.empty_cache()
        m_distance = np.round(np.asarray(m_distance),6)

        for i in range(r):
            for j in range(r):
                m_distance[i][j] = 1 - m_distance[i][j]
                if i!=j and m_distance[i][j] < 0.000001:
                    m_distance[i][j]= 0.000001
                elif i == j:
                    m_distance[i][j]=0
    else:
        m_distance = distance_matrix

    return get_mst_stats_and_diameter(m_distance)

def get_mst_stats_and_diameter(distance_matrix):
    # Compute MST
    graph = csr_matrix(distance_matrix)
    mst_sparse = minimum_spanning_tree(graph)
    mst_coo = mst_sparse.tocoo()
    n = distance_matrix.shape[0]
    
    # 1. Edge weights & basic stats
    weights = mst_coo.data
    total_weight = weights.sum()
    mean_w = weights.mean() if len(weights) > 0 else 0
    var_w = weights.var() if len(weights) > 0 else 0
    std_w = weights.std() if len(weights) > 0 else 0
    max_w = weights.max() if len(weights) > 0 else 0
    min_w = weights.min() if len(weights) > 0 else 0
    ratio_max_to_total = max_w / total_weight if total_weight > 0 else None
    
    # 2. Node degrees
    degrees = np.zeros(n, dtype=int)
    # build adjacency list with weights
    adj = defaultdict(list)
    for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        # Add both directions since it's undirected
        adj[u].append((v, w))
        adj[v].append((u, w))
        degrees[u] += 1
        degrees[v] += 1
    
    n_leaf = (degrees == 1).sum()
    # 3. Longest path (diameter) in weighted tree
    def bfs_farthest(start):
        # returns (farthest_node, distance, parent_map)
        visited = [False]*n
        dist = [0.0]*n
        parent = [-1]*n
        q = deque([start])
        visited[start] = True
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dist[v] = dist[u] + w
                    parent[v] = u
                    q.append(v)
        farthest = int(np.argmax(dist))
        return farthest, dist[farthest], parent
    
    # First BFS from node 0 (or any)
    node_a, _, _ = bfs_farthest(0)
    # Second BFS from node_a
    node_b, diameter_length, parent = bfs_farthest(node_a)
    
    # Reconstruct path from node_b back to node_a
    path = []
    cur = node_b
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path = path[::-1]  # reverse to start → end
    
    return total_weight, mean_w, max_w, min_w, std_w, degrees.mean(), degrees.max(), n_leaf, diameter_length, len(path)