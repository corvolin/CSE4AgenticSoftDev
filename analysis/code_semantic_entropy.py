import math
import torch
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN

from analysis.utils import get_noise_as_new_label, get_entropy, check_cluster_equivalence
from analysis.utils import get_thershold_by_reference, compute_mst_threshold_pruning_label, merge_cluster_labels, distance_matrix_to_mst, mst_pruning_to_stable_weight

# input a list of texts or a distance matrix and a minimum cluster size
# output HDBSCAN of the input set
def compute_semantic_entropy_HDBSCAN(texts, model=None, distance_matrix=None, min_cluster_size=2):
    if distance_matrix is None:
        m_distance = model.similarity(model.encode(texts), model.encode(texts))
        torch.cuda.empty_cache()
        m_distance = np.round(np.asarray(m_distance),6)

        r = len(texts)
        for i in range(r):
            for j in range(r):
                m_distance[i][j] = 1 - m_distance[i][j]
                if i!=j and m_distance[i][j] < 0.000001:
                    m_distance[i][j]= 0.000001
                elif i == j:
                    m_distance[i][j]=0
    else:
        m_distance = distance_matrix

    clustering = HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed', allow_single_cluster=True).fit(m_distance)

    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_))

# input a list of texts or a distance matrix and a reachability ranage
# output DBSCAN of the input set
def compute_semantic_entropy_DBSCAN(texts, model=None, distance_matrix=None, epsilon=0.2):
    if distance_matrix is None:
        m_distance = model.similarity(model.encode(texts), model.encode(texts))
        torch.cuda.empty_cache()
        m_distance = np.round(np.asarray(m_distance),6)

        r = len(texts)
        for i in range(r):
            for j in range(r):
                m_distance[i][j] = 1 - m_distance[i][j]
                if i!=j and m_distance[i][j] < 0.000001:
                    m_distance[i][j]= 0.000001
                elif i == j:
                    m_distance[i][j]=0
    else:
        m_distance = distance_matrix

    clustering = DBSCAN(eps=epsilon, min_samples=2, metric='precomputed').fit(distance_matrix)

    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_))

def compute_semantic_entropy_threshold(texts, threshold, model=None, distance_matrix=None):
    if distance_matrix is None:
        m_distance = model.similarity(model.encode(texts), model.encode(texts))
        torch.cuda.empty_cache()
        m_distance = np.round(np.asarray(m_distance),6)

        r = len(texts)
        for i in range(r):
            for j in range(r):
                m_distance[i][j] = 1 - m_distance[i][j]
                if i!=j and m_distance[i][j] < 0.000001:
                    m_distance[i][j]= 0.000001
                elif i == j:
                    m_distance[i][j]=0
    else:
        m_distance = distance_matrix

    clustering = DBSCAN(eps=threshold, min_samples=2, metric='precomputed').fit(m_distance)
    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_))

