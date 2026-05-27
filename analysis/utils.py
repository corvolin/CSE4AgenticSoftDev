import os
import json
import re
import math
import time
import numpy as np
from numpy import mean
import copy
import torch
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from generation.utils import extract_all_comments, extract_before_fences

from scipy.sparse import csr_matrix
from scipy.stats import kurtosis
from scipy.stats import skew

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict, deque
def add_prefix_suffix(row, l_base, prefix, l_suffix):

    for base in l_base:
        for suffix in l_suffix:
            row.append(prefix+base+'_'+suffix)
    
    row.append(prefix+'se_HDBSCAN_time')
    row.append(prefix+'se_scaled_time')
    row.append(prefix+'MST_time')
    

def get_column(model_keys, multi_agent='none'):
    l_method = ['dist', 'align', 'se', 'MST']
    l_se_function = ['HDBSCAN', 'linear', 'log', 'square', 'sigmoid', 'logit', 'linear_inverse', 'log_inverse', 'square_inverse', 'sigmoid_inverse', 'logit_inverse']
    l_variant = ['mean', 'max', 'min', 'std', 'time']
    l_topological = ['weight_sum', 'weight_mean', 'weight_max','weight_min', 'weight_std', 'degree_mean', 'degree_max', 'leaf_count', 'diameter_weight', 'diameter_count']

    l_measure = []

    for m in l_method:
        if m == 'se':
            for s in l_se_function:
                l_measure.append(m+'_'+s)
            
        elif m == 'MST':
            for t in l_topological:
                l_measure.append(m+'_'+t)

        else:
            for v in l_variant:
                l_measure.append(m+'_'+v)

    row = ['task_id']
    
    if 'funcAnalyst' in multi_agent:
        add_prefix_suffix(row, l_measure, 'func_plan_', model_keys)

    if 'coder' in multi_agent:
        add_prefix_suffix(row, l_measure, 'code_', model_keys)

    if 'funcReviewer' in multi_agent:
        add_prefix_suffix(row, l_measure, 'func_review_', model_keys)

    return row

def add_item_to_list(l_base, item, prefix, suffix):
    if isinstance(prefix, list):
        for p in prefix:
            if isinstance(suffix, list):
                for s in suffix:
                    l_base.append(p+'_'+item+'_'+s)
            else:
                l_base.append(p+'_'+item+'_'+suffix)
    else:
        if isinstance(suffix, list):
            for s in suffix:
                l_base.append(prefix+'_'+item+'_'+s)
            else:
                l_base.append(prefix+'_'+item+'_'+suffix)
    return l_base


def get_thershold_by_reference(base, target, reference, func='linear'):

    if func=='linear':
        threshold = target/base * reference
    elif func=='linear_inverse':
        threshold = (1 - target/base) * reference
    elif func=='log':
        threshold = math.log(target+1)/math.log(base+1) * reference
    elif func=='log_inverse':
        threshold = (1 - math.log(target+1)/math.log(base+1)) * reference
    elif func=='square':
        threshold = math.pow(target,2)/math.pow(base,2) * reference
    elif func=='square_inverse':
        threshold = (1 - math.pow(target,2)/math.pow(base,2)) * reference
    elif func=='sigmoid':
        threshold = (1 / (1 + np.exp(-((20 * target/base) - 10)))) * reference
    elif func=='sigmoid_inverse':
        threshold = (1 - (1 / (1 + np.exp(-((20 * target/base) - 10))))) * reference
    elif 'logit' in func:
        p = target/base
        if target == 0:
            p = 0.0001
        elif target>base:
            p = 0.9999
        if 'inverse' in func:
            threshold = (1 - math.log(p / (1 - p))/10+0.5) * reference
        else:
            threshold = (math.log(p / (1 - p))/10+0.5) * reference

    if threshold > reference:
        return reference
    elif threshold < reference*0.01:
        return reference*0.01
    else:
        return threshold

def get_list_code_threshold(model, thresholds):
    mt = thresholds[model]
    return [mt['code_dist'][10], mt['code_dist'][25], mt['code_dist'][50], mt['code_dist'][75], mt['new_code_dist'][10], mt['new_code_dist'][25], mt['new_code_dist'][50], mt['new_code_dist'][75]]

def get_list_func_plan_threshold(model, thresholds):
    mt = thresholds[model]
    return [mt['func_plan_dist'][10], mt['func_plan_dist'][25], mt['func_plan_dist'][50], mt['func_plan_dist'][75]]


def check_cluster_equivalence(texts, target_text, model=None):
    for text in texts:
        if not check_bidirectional_equivalence(text, target_text, model):
            return False
    return True

def check_bidirectional_equivalence(text1, text2, model=None):
    model.history_message_clear()
    reviewer_func_res = model.task_review([text1, text2], topic='func_eq')
    model.history_message_clear()
    review = extract_before_fences(reviewer_func_res).strip()
    review = extract_before_fences(review,fence='\n\n').strip()
    forward_eq = False
    if 'yes' in review or 'Yes' in review or 'YES' in review:
        forward_eq = True

    reviewer_func_res = model.task_review([text2, text1], topic='func_eq')
    model.history_message_clear()
    review = extract_before_fences(reviewer_func_res).strip()
    review = extract_before_fences(review,fence='\n\n').strip()
    backward_eq = False
    if 'yes' in review or 'Yes' in review or 'YES' in review:
        backward_eq = True
    if forward_eq and backward_eq:
        return True
    return False

def get_entropy(labels):
    pos_lables = [l + 1 for l in labels]
    ps = np.bincount(pos_lables) / len(pos_lables)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def get_noise_as_new_label(labels):
    max_label = max(labels)
    denoise = [l for l in labels if l >= 0]
    for i in range(len(labels)-len(denoise)):
        denoise.append(max_label+i+1)
    return denoise

def extract_corpus(data_dir, doc_type):
    doc_corpus = []
    for vul in os.listdir(data_dir):
        if 'report' in vul:
            continue
        for scenario in os.listdir(os.path.join(data_dir,vul)):
            if doc_type == 'base':
                base_plan_file = os.path.join('data','vulnerability',vul,scenario,'func_context.py')
                with open(base_plan_file, 'r') as f:
                    data = f.read().rstrip()
                    data = ' '.join(preprocess_sentence(extract_all_comments(data)))
                    doc_corpus.append(data)
            else:
                doc_dir = os.path.join(data_dir,vul,scenario,doc_type)
                stat_file = os.path.join(data_dir,vul,scenario,'stat.json')

                with open(stat_file, 'r') as f:
                    stat = json.load(f) 
                for module in stat:
                    doc_file = os.path.join(doc_dir,module.replace('py','json'))
                    with open(doc_file, 'r') as f:
                        data =  f.read().rstrip()
                        data = ' '.join(preprocess_sentence(data))
                        doc_corpus.append(data)
    return doc_corpus

def compute_plan_entropy(data_dir, corpus_probs):
    plan_entropy_total = []
    plan_entropy_avg = []
    
    for vul in os.listdir(data_dir):
        if 'report' in vul:
            continue
        for scenario in os.listdir(os.path.join(data_dir,vul)):
            plan_dir = os.path.join(data_dir,vul,scenario,'plan')
            stat_file = os.path.join(data_dir,vul,scenario,'stat.json')

            with open(stat_file, 'r') as f:
                stat = json.load(f) 
            for module in stat:
                plan_file = os.path.join(plan_dir,module.replace('py','json'))
                with open(plan_file, 'r') as f:
                    data =  f.read().rstrip()
                    data = preprocess_sentence(data)
                    total, avg = sentence_entropy(data, corpus_probs)
                    plan_entropy_total.append(total)
                    plan_entropy_avg.append(avg)
    return plan_entropy_total, plan_entropy_avg

# Compute sentence entropy
def sentence_entropy(sentence_words, probs, base=2):
    total = 0.0
    for w in sentence_words:
        p = probs.get(w.lower(), 1e-12)
        total += -math.log(p, base)
    avg = total / len(sentence_words)
    return total, avg

def simple_json_repair(old_str):
    new_str = old_str.replace(' None ', '"None"')
    new_str = new_str.replace(' Ok ', '"Ok"')
    new_str = new_str.replace('""', '"')
    return new_str



def compute_specificity_entropy(data_dir, corpus):
    specificity_entropy = []
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    word2idf = dict(zip(vectorizer.get_feature_names_out(), idf))
    
    
    for vul in os.listdir(data_dir):
        if 'report' in vul:
            continue
        for scenario in os.listdir(os.path.join(data_dir,vul)):
            plan_dir = os.path.join(data_dir,vul,scenario,'plan')
            stat_file = os.path.join(data_dir,vul,scenario,'stat.json')

            with open(stat_file, 'r') as f:
                stat = json.load(f) 
            for module in stat:
                plan_file = os.path.join(plan_dir,module.replace('py','json'))
                with open(plan_file, 'r') as f:
                    data =  f.read().rstrip()
                    words = preprocess_sentence(data)
                    freqs = {word: words.count(word)/len(words) for word in set(words)}
                    specificity_entropy.append(-sum(f * np.log2(f) * word2idf.get(word, 0) for word, f in freqs.items()))
    return specificity_entropy

def compute_specificity_bert(data_dir, corpus):
    specificity_bert = []
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    word2idf = dict(zip(vectorizer.get_feature_names_out(), idf))
    
    for vul in os.listdir(data_dir):
        if 'report' in vul:
            continue
        for scenario in os.listdir(os.path.join(data_dir,vul)):
            plan_dir = os.path.join(data_dir,vul,scenario,'plan')
            stat_file = os.path.join(data_dir,vul,scenario,'stat.json')

            base_plan_file = os.path.join('data','vulnerability',vul,scenario,'func_context.py')
            with open(base_plan_file, 'r') as f:
                base_data = f.read().rstrip()
                tokens = preprocess_sentence(extract_all_comments(base_data))
                filtered = [t for t in tokens if t.lower() in word2idf]
                base_data = " ".join(filtered)

            with open(stat_file, 'r') as f:
                stat = json.load(f) 
            for module in stat:
                plan_file = os.path.join(plan_dir,module.replace('py','json'))
                with open(plan_file, 'r') as f:
                    data =  f.read().rstrip()

                    tokens = preprocess_sentence(data)
                    filtered = [t for t in tokens if t.lower() in word2idf]
                    data = " ".join(filtered)

                    score([data],[base_data],lang='en',idf=word2idf)[2].item()
    return specificity_bert

def compute_meteor(data_dir):
    l_meteor_score = []

    for vul in os.listdir(data_dir):
        if 'report' in vul:
            continue
        for scenario in os.listdir(os.path.join(data_dir,vul)):
            plan_dir = os.path.join(data_dir,vul,scenario,'plan')
            stat_file = os.path.join(data_dir,vul,scenario,'stat.json')

            base_plan_file = os.path.join('data','vulnerability',vul,scenario,'func_context.py')
            with open(base_plan_file, 'r') as f:
                base_data = f.read().rstrip()
                base_data = preprocess_sentence(extract_all_comments(base_data))

            with open(stat_file, 'r') as f:
                stat = json.load(f) 
            for module in stat:
                plan_file = os.path.join(plan_dir,module.replace('py','json'))
                with open(plan_file, 'r') as f:
                    data =  f.read().rstrip()
                    data = preprocess_sentence(data)

                    l_meteor_score.append(meteor_score([base_data], data))
    return l_meteor_score

def compute_self_BLEU(data_dir):
    self_BLEU = []
    
    for vul in os.listdir(data_dir):
        if 'report' in vul:
            continue
        for scenario in os.listdir(os.path.join(data_dir,vul)):
            plan_dir = os.path.join(data_dir,vul,scenario,'plan')
            stat_file = os.path.join(data_dir,vul,scenario,'stat.json')

            with open(stat_file, 'r') as f:
                stat = json.load(f) 
            for module in stat:
                plan_file = os.path.join(plan_dir,module.replace('py','json'))
                try:
                    with open(plan_file, 'r') as f:
                        plan = json.load(f)
                    self_BLEU.append(calculate_selfBleu(extract_terminal_pairs(plan)))
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON: {e}",plan_file)
                    self_BLEU.append(0)
    return self_BLEU

def preprocess_sentence(text: str) -> list[str]:
    data = re.sub(r'\\n|\\t', ' ', text)
    # 1. Remove newline and tab characters
    data = re.sub(r'[\r\n\t]+', '', data)
    # 2. Remove all digits
    data = re.sub(r'\d+', ' ', data)
    # 3. Remove special characters (keep only letters and spaces)
    data = re.sub(r'[^A-Za-z\s]+', ' ', data)
    # Collapse multiple spaces and trim
    data = re.sub(r'\s+', ' ', data).lower().strip()
    
    wnl = WordNetLemmatizer()
    lemmatized = []
    for word in data.split():
        lemmatized.append(wnl.lemmatize(word, pos="v"))
    if 'treturn' in lemmatized:
        print("---------------")
        print(text)
        print(data)
    return lemmatized


def extract_terminal_pairs(obj, parent_key=''):
    result = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key} {k}".strip()
            if isinstance(v, (dict, list)):
                result.extend(extract_terminal_pairs(v, new_key))
            else:
                result.append(f"{new_key} {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                result.extend(extract_terminal_pairs(item, parent_key))
            else:
                result.append(f"{parent_key} {item}".strip())
    return result

def calculate_sentence_selfBleu(sentence, remaining_sentences):
    lst = []
    for i in remaining_sentences:
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([sentence], i, smoothing_function=smoothing)
        lst.append(bleu)
    return lst


def calculate_selfBleu(sentences):
    '''
    sentences - list of sentences generated by NLG system
    '''
    bleu_scores = []

    for i in sentences:
        sentences_copy = copy.deepcopy(sentences)
        remaining_sentences = sentences_copy.remove(i)
        bleu = calculate_sentence_selfBleu(i,sentences_copy)
        bleu_scores.append(bleu)
    if math.isnan(mean(bleu_scores)):
        return 0
    return mean(bleu_scores)

def extract_requirement(data, task, dataset):
    for row in data:
        if row['task_id'] == task:
            if dataset == 'HumanEval.jsonl':
                requirement = extract_all_comments(row['prompt'])
            elif dataset == 'bigCodeBench_hard.jsonl':
                requirement = row['instruct_prompt_clean']
            elif dataset == 'codeguard_python.jsonl':
                requirement = ''
            return requirement
        
def majority_true(flags):
    """
    Returns True if strictly more than half of the values are True.
    """
    if not flags:
        return False  # or raise ValueError("Empty list")

    return sum(flags) > len(flags) / 2

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

def distance_matrix_to_mst(distance_matrix):
    """
    Convert a distance matrix to a Minimum Spanning Tree (MST).

    Parameters
    ----------
    distance_matrix : (n, n) array-like

    Returns
    -------
    mst_edges : list of tuples (u, v, weight)
    mst_matrix : (n, n) numpy array (adjacency matrix of MST)
    """

    D = np.asarray(distance_matrix)
    n = D.shape[0]

    # Step 1: Build edge list (upper triangle only)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((D[u, v], u, v))

    # Step 2: Sort edges by weight
    edges.sort(key=lambda x: x[0])

    # Step 3: Kruskal's algorithm
    uf = UnionFind(n)
    mst_edges = []

    for w, u, v in edges:
        if uf.union(u, v):
            mst_edges.append((u, v, w))
        if len(mst_edges) == n - 1:
            break

    # Step 4: Build MST adjacency matrix
    mst_matrix = np.zeros((n, n))
    for u, v, w in mst_edges:
        mst_matrix[u, v] = w
        mst_matrix[v, u] = w

    return mst_edges, mst_matrix

def mst_to_qunatiles(mst_edges):
    if not mst_edges:
        return np.nan

    weights = np.array([w for _, _, w in mst_edges])

    if len(weights) < 2:
        return 0.0
    
    q10 = np.percentile(weights, 10)
    q90 = np.percentile(weights, 90)
    q90_q10 = q90 - q10

    return q90_q10

def mst_to_kurtosis(mst_edges):
    if not mst_edges:
        return np.nan

    weights = np.array([w for _, _, w in mst_edges])

    if len(weights) < 2:
        return np.nan
    
    mean = weights.mean()
    std = weights.std()

    if std == 0:
        return 0.0
    return kurtosis(weights, fisher=True, bias=False)
    

def mst_to_max_min(mst_edges):
    if not mst_edges:
        return np.nan, np.nan

    weights = np.array([w for _, _, w in mst_edges])

    return max(weights), min(weights)

def mst_pruning_to_stable_weight(mst_edges, criteria, threshold):
    if not mst_edges:
        return np.nan

    if not (0 < threshold < 1):
        raise ValueError("threshold must be between 0 and 1")

    if criteria not in ("mean", "std", "both"):
        raise ValueError("flag must be 'mean' or 'std' or both")

    # Extract weights sorted descending
    weights = np.array(sorted([w for _, _, w in mst_edges], reverse=True))

    # Original statistic (fixed reference)

    # Current state
    current_weights = weights.copy()
    current_mean = weights.mean()
    current_std = weights.std()

    # Iterative pruning
    for i in range(len(weights)):
        if len(current_weights) <= 1:
            break

        # Remove current largest
        new_weights = current_weights[1:]  # since sorted descending

        new_mean = new_weights.mean()
        new_std = new_weights.std()

        # Check condition

        if criteria=='mean' and new_mean < threshold * current_mean:
            current_weights = new_weights
            current_mean = current_weights.mean()
            current_std = current_weights.std()

        elif criteria=='std' and new_std < threshold * current_std:
            current_weights = new_weights
            current_mean = current_weights.mean()
            current_std = current_weights.std()

        elif criteria=='both' and ((new_std < threshold * current_std) or (new_mean < threshold * current_mean)):
            current_weights = new_weights
            current_mean = current_weights.mean()
            current_std = current_weights.std()
        else:
            break

    # Return largest remaining edge
    #print('mst to stable', criteria, threshold, current_weights[0])
    return current_weights

def edge_type_stats(distance_matrix, labels):
    """
    Parameters
    ----------
    distance_matrix : (n, n) array-like
        Symmetric pairwise distance matrix
    labels : list or array-like of length n
        Node labels: either 'white'/'black' OR boolean (True=white, False=black)

    Returns
    -------
    dict with keys:
        'white-white', 'black-black', 'white-black'
        each containing {'mean': float, 'std': float, 'count': int}
    """

    D = np.asarray(distance_matrix)
    n = D.shape[0]

    # Normalize labels to boolean: True = white, False = black
    labels = np.asarray(labels)
    if labels.dtype != bool:
        labels = (labels == "white")

    # Upper triangle indices (exclude diagonal, avoid duplicates)
    i, j = np.triu_indices(n, k=1)

    dists = D[i, j]
    li = labels[i]
    lj = labels[j]

    # Masks for edge types
    ww_mask = (li & lj)
    bb_mask = (~li & ~lj)
    wb_mask = (li ^ lj)
    
    def stats(arr):
        if arr.size == 0:
            return {"mean": np.nan, "median":np.nan, "std": np.nan, "count": 0, "quantile":np.nan, "kurtosis":np.nan, "skew": np.nan, "peak_count": 0, "max_mode": 0, "min_mode":0}
        if np.var(arr)==0:
            return {"mean": float(arr.mean()), "median": float(np.median(arr)),"std": float(arr.std()), "count": int(arr.size), "quantile": 0, "kurtosis":-2, "skew": 0, "peak_count": 1, "max_mode": float(arr.mean()), "min_mode": float(arr.mean())}
        kde = gaussian_kde(arr)
        xs = np.linspace(min(arr), max(arr), 1000)
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
            
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "count": int(arr.size),
            "quantile": float(np.percentile(arr, 90)-np.percentile(arr, 10)), 
            "kurtosis": kurtosis(arr, fisher=True, bias=False),
            "skew": skew(arr),
            "peak_count": peak_count,
            "max_mode": max_mode,
            "min_mode": min_mode
        }

    global_stats = {
        #"pass_pass": stats(dists[ww_mask]),
        #"fail_fail": stats(dists[bb_mask]),
        #"pass_fail": stats(dists[wb_mask]),
        "all": stats(dists)
    }

 # ---------- PART 2: MST (Kruskal) ----------

    start = time.perf_counter() 
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((D[u, v], u, v))

    edges.sort(key=lambda x: x[0])

    uf = UnionFind(n)
    mst_edges = []

    for w, u, v in edges:
        if uf.union(u, v):
            mst_edges.append((w, u, v))
        if len(mst_edges) == n - 1:
            break
    end = time.perf_counter() 

    # ---------- PART 3: MST EDGE TYPE STATS ----------
    mst_groups = {
        "pass_pass": [],
        "fail_fail": [],
        "pass_fail": [],
        "all": []
    }

    for w, u, v in mst_edges:
        if labels[u] and labels[v]:
            mst_groups["pass_pass"].append(w)
        elif not labels[u] and not labels[v]:
            mst_groups["fail_fail"].append(w)
        else:
            mst_groups["pass_fail"].append(w)
        mst_groups["all"].append(w)

    mst_stats = {
        #k: stats(np.array(v)) for k, v in mst_groups.items()
        "all": stats(np.array(mst_groups["all"]))
    }

    return {
        "global": global_stats,
        "mst": mst_stats,
        "mst_time": start-end
    }

def compute_mst_threshold_pruning_label(distance_matrix, criteria='mean', threshold=0.9):
    """
    Parameters
    ----------
    distance_matrix : (n, n) array-like
    threshold : float in (0, 1)

    Returns
    -------
    labels : list[int]
        Cluster labels (0..k-1)
    """

    if not (0 < threshold < 1):
        raise ValueError("threshold must be between 0 and 1")

    D = np.asarray(distance_matrix)
    n = D.shape[0]

    # ---------- Step 1: Build MST (Kruskal) ----------
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((D[u, v], u, v))

    edges.sort(key=lambda x: x[0])

    uf = UnionFind(n)
    mst_edges = []

    for w, u, v in edges:
        if uf.union(u, v):
            mst_edges.append((w, u, v))
        if len(mst_edges) == n - 1:
            break

    # ---------- Step 2: Sort edges descending ----------
    mst_edges.sort(reverse=True, key=lambda x: x[0])

    # Track current edges
    current_edges = mst_edges.copy()
    current_weights = np.array([w for w, _, _ in current_edges])

    current_mean = current_weights.mean()
    current_std = current_weights.std()

    # ---------- Step 3: Iterative pruning ----------
    for edge in mst_edges:
        w, u, v = edge

        # Try removing this edge
        temp_edges = [e for e in current_edges if e != edge]

        if len(temp_edges) == 0:
            break

        temp_weights = np.array([e[0] for e in temp_edges])
        
        temp_mean = temp_weights.mean()
        temp_std = temp_weights.std()


        # Check stopping condition
        
        if criteria=='mean' and temp_mean < threshold*current_mean:
            # Accept removal
            current_edges = temp_edges
            current_mean = temp_mean
            current_std = temp_std
        elif criteria=='std' and temp_std < threshold*current_std:
            # Accept removal
            current_edges = temp_edges
            current_mean = temp_mean
            current_std = temp_std
        elif criteria=='both' and ((temp_mean < threshold*current_mean)or(temp_std < threshold*current_std)):
            # Accept removal
            current_edges = temp_edges
            current_mean = temp_mean
            current_std = temp_std
        else:
            # Stop immediately (greedy stopping)
            break
    #print('mst pruning', criteria, threshold, current_threshold)
    # ---------- Step 4: Build clusters ----------
    uf = UnionFind(n)
    for w, u, v in current_edges:
        uf.union(u, v)

    # Assign compact labels
    root_to_label = {}
    labels = [0] * n
    label_id = 0

    for i in range(n):
        root = uf.find(i)
        if root not in root_to_label:
            root_to_label[root] = label_id
            label_id += 1
        labels[i] = root_to_label[root]

    return labels


def merge_cluster_labels(labels_a, labels_b, flag):
    """
    Parameters
    ----------
    labels_a : list[int]
    labels_b : list[int]
    flag : str ('union' or 'intersection')

    Returns
    -------
    merged_labels : list[int]
    """

    if len(labels_a) != len(labels_b):
        raise ValueError("Input label lists must have same length")

    n = len(labels_a)

    if flag == "union":
        # --- Build bipartite connectivity via Union-Find ---
        uf = UnionFind(n)

        # Map cluster → nodes
        clusters_a = defaultdict(list)
        clusters_b = defaultdict(list)

        for i in range(n):
            clusters_a[labels_a[i]].append(i)
            clusters_b[labels_b[i]].append(i)

        # For each cluster in A, union all nodes that share a B cluster
        # Efficient trick: group by (A label, B label)
        pair_groups = defaultdict(list)
        for i in range(n):
            pair_groups[(labels_a[i], labels_b[i])].append(i)

        # Connect nodes that share either A or B cluster via overlap
        for nodes in pair_groups.values():
            base = nodes[0]
            for node in nodes[1:]:
                uf.union(base, node)

        # Additionally connect within same A cluster
        for nodes in clusters_a.values():
            base = nodes[0]
            for node in nodes[1:]:
                uf.union(base, node)

        # And within same B cluster
        for nodes in clusters_b.values():
            base = nodes[0]
            for node in nodes[1:]:
                uf.union(base, node)

        # Assign labels
        root_to_label = {}
        merged = [0] * n
        label_id = 0

        for i in range(n):
            r = uf.find(i)
            if r not in root_to_label:
                root_to_label[r] = label_id
                label_id += 1
            merged[i] = root_to_label[r]

        return merged

    elif flag == "intersection":
        # --- Exact refinement: clusters = intersections of A and B ---
        pair_groups = defaultdict(list)

        for i in range(n):
            pair_groups[(labels_a[i], labels_b[i])].append(i)

        merged = [0] * n
        label_id = 0

        for nodes in pair_groups.values():
            for i in nodes:
                merged[i] = label_id
            label_id += 1

        return merged

    else:
        raise ValueError("flag must be 'union' or 'intersection'")