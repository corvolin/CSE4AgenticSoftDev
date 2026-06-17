import torch
import math 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
from collections import defaultdict, deque

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN

from analysis.utils import get_noise_as_new_label, get_entropy, check_cluster_equivalence
from analysis.utils import compute_mst_threshold_pruning_label, merge_cluster_labels, distance_matrix_to_mst, mst_pruning_to_stable_weight


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



def compute_mst_pruning_entropy(distance_matrix, criteria, threshold):
    labels = compute_mst_threshold_pruning_label(distance_matrix, criteria, threshold)
    return get_entropy(get_noise_as_new_label(labels))/math.log2(len(labels)), get_noise_as_new_label(labels)


def compute_joint_cluter_entropy(label_1, label_2, operation):

    labels = merge_cluster_labels(label_1, label_2, operation)

    return get_entropy(get_noise_as_new_label(labels))/math.log2(len(labels))

def compute_mst_mean_threshold_DBSCAN_entropy(distance_matrix, threshold):
    mst_edges, mst_matrix = distance_matrix_to_mst(distance_matrix)
    weights = [w for _, _, w in mst_edges]
    mean_weight =  float(np.mean(weights)) if weights else 0.0
    if mean_weight==0:
        mean_weight=0.001

    #print('mean_threshold_DBSCAN', threshold, mean_weight*threshold)
    clustering = DBSCAN(eps=mean_weight*threshold, min_samples=2, metric='precomputed').fit(distance_matrix)
    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_)), get_noise_as_new_label(clustering.labels_)

def compute_mean_threshold_DBSCAN_entropy(distance_matrix, threshold):
    mean_weight =  float(np.mean(distance_matrix))
    if mean_weight==0:
        mean_weight=0.001

    #print('mean_threshold_DBSCAN', threshold, mean_weight*threshold)
    clustering = DBSCAN(eps=mean_weight*threshold, min_samples=2, metric='precomputed').fit(distance_matrix)
    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_)), get_noise_as_new_label(clustering.labels_)

def compute_mst_DBSCAN_entropy(distance_matrix, criteria, threshold):
    mst_edges, mst_matrix = distance_matrix_to_mst(distance_matrix)
    
    stable_weights = mst_pruning_to_stable_weight(mst_edges, criteria, threshold)
    largest_weight = stable_weights[0]

    if largest_weight==0:
        largest_weight=0.000001
    #print('mst_DBSCAN_entropy', criteria, threshold, largest_weight)
    clustering = DBSCAN(eps=largest_weight * 1.001, min_samples=2, metric='precomputed').fit(distance_matrix)
    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_)), get_noise_as_new_label(clustering.labels_)

def compute_ratio_DBSCAN_entropy(distance_matrix, criteria, threshold, use_mst=True):
    mst_edges, mst_matrix = distance_matrix_to_mst(distance_matrix)
    weights = [w for _, _, w in mst_edges]

    global_mean = float(np.mean(distance_matrix))
    mst_mean = float(np.mean(weights))

    if global_mean == 0:
        global_mean = 0.000001

    if use_mst:
        ref_target = weights
    else:
        ref_target = distance_matrix.flatten()

    ref_weight = 0.000001
    if criteria == 'mean':
        ref_weight = float(np.mean(ref_target))
    elif criteria == 'median':
        ref_weight = np.median(ref_target)

    elif 'mode' in criteria:
        if np.var(ref_target)==0:
            ref_weight = float(np.mean(ref_target))
        else:
            kde = gaussian_kde(ref_target)
            xs = np.linspace(np.min(ref_target), np.max(ref_target), 1000)
            density = kde(xs)
            peaks, _ = find_peaks(density)

            if len(peaks)>1:
                peak_count = len(peaks)
                max_mode = max(xs[peaks])
                min_mode = min(xs[peaks])
            else:
                peak_count = 1
                max_mode = xs[np.argmax(density)]
                min_mode = max_mode

            if criteria == 'max_mode':
                ref_weight = max_mode
            elif criteria == 'min_mode':
                ref_weight = min_mode

    ratio = 2*(1-(mst_mean/global_mean))*ref_weight
    if ratio <= 0:
        ratio = 0.001
        
    clustering = DBSCAN(eps=ratio*threshold, min_samples=2, metric='precomputed').fit(distance_matrix)
    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_)), get_noise_as_new_label(clustering.labels_)

def compute_mst_pruning_ratio_DBSCAN_entropy(distance_matrix, threshold):
    mst_edges, mst_matrix = distance_matrix_to_mst(distance_matrix)
    
    stable_weights = mst_pruning_to_stable_weight(mst_edges, 'both', 0.8)

    global_mean = float(np.mean(distance_matrix))
    mst_mean = float(np.mean(stable_weights))

    if global_mean == 0:
        global_mean = 0.000001

    ratio = (1-(mst_mean/global_mean))*mst_mean
    if ratio <= 0:
        ratio = 0.001
        
    clustering = DBSCAN(eps=ratio*threshold, min_samples=2, metric='precomputed').fit(distance_matrix)
    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_)), get_noise_as_new_label(clustering.labels_)

    
    