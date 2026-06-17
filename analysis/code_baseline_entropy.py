import math
import torch
from continuous_eval.metrics.code.python import PythonASTSimilarity
from analysis.utils import get_noise_as_new_label, get_entropy, check_cluster_equivalence

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