import numpy as np
import math
import torch

def compute_semantic_distance(texts, model=None):
    r = len(texts)
    m_distance = model.similarity(model.encode(texts), model.encode(texts))
    torch.cuda.empty_cache()

    for i in range(r):
        for j in range(r):
            m_distance[i][j] = 1 - m_distance[i][j]
            if i!=j and m_distance[i][j] < 0.0001:
                m_distance[i][j]= 0.0001
            elif i == j:
                m_distance[i][j]=0

    m_distance = np.round(np.asarray(m_distance),4)

    for i in range(r):
        for j in range(r):
            if i!=j and m_distance[i][j] != m_distance[j][i]:
                print(i,j, m_distance[i][j], m_distance[j][i])

                m_distance[j][i] = m_distance[i][j] 

    dist_mean, dist_max, dist_min, dist_std = get_stats_pairwise_distances(m_distance)
    return float(dist_mean), float(dist_max), float(dist_min), float(dist_std), m_distance


def get_stats_pairwise_distances(dist_matrix, condensed=False):
    if condensed:
        # Already contains only i<j pairs
        total_sum = dist_matrix.sum()
    else:
        # Full matrix includes i<j and i>j (and diagonal zeros).
        # We only sum i<j to avoid double counting
        # Use np.triu_indices to get upper triangle i<j
        n = dist_matrix.shape[0]
        iu = np.triu_indices(n, k=1)
        values = dist_matrix[iu]
    mean_dist = values.mean()
    max_dist = values.max()
    min_dist = values.min()
    std_dist  = values.std() 
    return float(mean_dist), float(max_dist), float(min_dist), float(std_dist)