import math
import torch
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from continuous_eval.metrics.code.python import PythonASTSimilarity

from analysis.utils import get_noise_as_new_label, get_entropy, check_cluster_equivalence
from analysis.utils import get_thershold_by_reference, compute_mst_threshold_pruning_label, merge_cluster_labels, distance_matrix_to_mst, mst_pruning_to_stable_weight

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


def compute_semantic_entropy_clustering(texts, model=None, threshold=None, base_value=None, target_value=None, reference_value=None, distance_matrix=None):
    l_se = []

    r = len(texts)
    if distance_matrix is None:
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
    labels = get_labels_from_equivalence(texts, model, metric='semantic')
    return get_entropy(get_noise_as_new_label(labels))/math.log2(len(labels)), get_noise_as_new_label(labels)

# group identical strings into same cluster
def compute_lexical_entropy_equivalence(texts):
    labels = get_labels_from_equivalence(texts, metric='lexical')
    return get_entropy(get_noise_as_new_label(labels))/math.log2(len(labels)), get_noise_as_new_label(labels)

# group identical AST structure into same cluster
def compute_structural_entropy_equivalence(texts):
    labels = get_labels_from_equivalence(texts, metric='structural')
    return get_entropy(get_noise_as_new_label(labels))/math.log2(len(labels)), get_noise_as_new_label(labels)

def get_labels_from_equivalence(texts, model=None, metric='lexical'):
    labels = dict()
    ast_metric = PythonASTSimilarity()

    for i in range(len(texts)):
        text = texts[i]
        if len(labels) == 0:
            labels[0] = [text]
        else:
            cluseter_found = False
            for key in labels:
                if metric=='semantic':
                    if check_cluster_equivalence(labels[key], text, model):
                        cluseter_found = True
                        labels[key].append(text)
                        break
                elif metric=='lexical':
                    if labels[key][0] == text:
                        cluseter_found = True
                        labels[key].append(text)
                        break
                elif metric=='structural':
                    datum = {
                        "answer": text,
                        "ground_truth_answers": [labels[key][0]]
                    }
                    if ast_metric(**datum)['Python_AST_Similarity']==1:
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

    
    