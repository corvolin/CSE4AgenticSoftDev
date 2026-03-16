import math
import torch
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN


from analysis.utils import get_noise_as_new_label, get_entropy, check_cluster_equivalence
from analysis.utils import get_thershold_by_reference

def compute_semantic_entropy_HDBSCAN(texts, model=None, distance_matrix=None):
    if distance_matrix is None:
        m_distance = model.similarity(model.encode(texts), model.encode(texts))
        torch.cuda.empty_cache()
        m_distance = np.round(np.asarray(m_distance),4)

        r = len(texts)
        for i in range(r):
            for j in range(r):
                m_distance[i][j] = 1 - m_distance[i][j]
                if i!=j and m_distance[i][j] < 0.0001:
                    m_distance[i][j]= 0.0001
                elif i == j:
                    m_distance[i][j]=0
    else:
        m_distance = distance_matrix

    clustering = HDBSCAN(min_cluster_size=2, metric='precomputed', allow_single_cluster=True).fit(m_distance)

    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_))

def compute_semantic_entropy_threshold(texts, threshold, model=None, distance_matrix=None):
    if distance_matrix is None:
        m_distance = model.similarity(model.encode(texts), model.encode(texts))
        torch.cuda.empty_cache()
        m_distance = np.round(np.asarray(m_distance),4)

        r = len(texts)
        for i in range(r):
            for j in range(r):
                m_distance[i][j] = 1 - m_distance[i][j]
                if i!=j and m_distance[i][j] < 0.0001:
                    m_distance[i][j]= 0.0001
                elif i == j:
                    m_distance[i][j]=0
    else:
        m_distance = distance_matrix

    clustering = DBSCAN(eps=threshold, min_samples=2, metric='precomputed').fit(m_distance)
    return get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_))


def compute_semantic_entropy_clustering(texts, model=None, threshold=None, base_value=None, target_value=None, reference_value=None, distance_matrix=None):
    l_se = []

    r = len(texts)
    if distance_matrix is None:
        m_distance = model.similarity(model.encode(texts), model.encode(texts))
        torch.cuda.empty_cache()
        m_distance = np.round(np.asarray(m_distance),4)

        for i in range(r):
            for j in range(r):
                m_distance[i][j] = 1 - m_distance[i][j]
                if i!=j and m_distance[i][j] < 0.0001:
                    m_distance[i][j]= 0.0001
                elif i == j:
                    m_distance[i][j]=0
    else:
        m_distance = distance_matrix

    if threshold != None:
        clustering = HDBSCAN(min_cluster_size=2, metric='precomputed', allow_single_cluster=True).fit(m_distance)
        l_result = []
        l_result.append(get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_)))

        return l_result
    
    else:
        l_threshold_func = ['logit', 'logit_inverse']
        l_se = []

        for threshold_func in l_threshold_func:
            threshold = get_thershold_by_reference(base_value, target_value, reference_value, threshold_func)  
            clustering = DBSCAN(eps=threshold, min_samples=2, metric='precomputed').fit(m_distance)
            l_se.append(get_entropy(get_noise_as_new_label(clustering.labels_))/math.log2(len(clustering.labels_)))

        return l_se


# groupe string with LLM bi-directional equivalence to same cluster
def compute_semantic_entropy_equivalence(texts, model=None):
    labels = get_labels_from_equivalence(texts, model)
    return get_entropy(get_noise_as_new_label(labels))/math.log2(len(labels))

def get_labels_from_equivalence(texts, model=None):
    labels = dict()
    for i in range(len(texts)):
        text = texts[i]
        if len(labels) == 0:
            labels[0] = [text]
        else:
            cluseter_found = False
            for key in labels:
                if check_cluster_equivalence(labels[key], text, model):
                    cluseter_found = True
                    labels[key].append(text)
                    break
            if not cluseter_found:
                labels[max(labels.keys())+1]=[text]

    l_labels = []
    for key in labels:
        for item in labels[key]:
            l_labels.append(key)

    return l_labels

# group identical strings into same cluster
def compute_semantic_entropy_identical(texts):
    labels = get_labels_from_identical(texts)
    return get_entropy(get_noise_as_new_label(labels))/math.log2(len(labels))

def get_labels_from_identical(texts):
    labels = dict()
    for i in range(len(texts)):
        text = texts[i]
        if len(labels) == 0:
            labels[0] = [text]
        else:
            cluseter_found = False
            for key in labels:
                if labels[key][0] == text:
                    cluseter_found = True
                    labels[key].append(text)
                    break
            if not cluseter_found:
                labels[max(labels.keys())+1]=[text]

    l_labels = []
    for key in labels:
        for item in labels[key]:
            l_labels.append(key)

    return l_labels
