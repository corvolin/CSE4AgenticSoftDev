import numpy as np
import math
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from continuous_eval.metrics.code.python import PythonASTSimilarity
from scipy.stats import kurtosis

def compute_lexical_distance(texts):
    n = len(texts)
    D = np.zeros((n, n))

    # Tokenize (simple whitespace; replace with better tokenizer if needed)
    tokenized = [t.split() for t in texts]

    smoothing = SmoothingFunction().method1

    for i in range(n):
        for j in range(i + 1, n):
            ref_i = tokenized[i]
            ref_j = tokenized[j]

            # BLEU(i -> j) and BLEU(j -> i)
            bleu_ij = sentence_bleu([ref_j], ref_i, smoothing_function=smoothing)
            bleu_ji = sentence_bleu([ref_i], ref_j, smoothing_function=smoothing)

            bleu_sym = 0.5 * (bleu_ij + bleu_ji)

            dist = 1.0 - bleu_sym

            D[i, j] = dist
            D[j, i] = dist

    dist_mean, dist_max, dist_min, dist_std = get_stats_pairwise_distances(D)
    return float(dist_mean), float(dist_max), float(dist_min), float(dist_std),D

def compute_structural_distance(texts):
    n = len(texts)
    D = np.zeros((n, n))
    ast_metric = PythonASTSimilarity()


    for i in range(n):
        for j in range(i + 1, n):
            datum = {
                "answer": texts[i],
                "ground_truth_answers": [texts[j]]
            }
            ast_ij = ast_metric(**datum)['Python_AST_Similarity']
            datum = {
                "answer": texts[j],
                "ground_truth_answers": [texts[i]]
            }
            ast_ji = ast_metric(**datum)['Python_AST_Similarity']

            ast_sym = 0.5 * (ast_ij + ast_ji)

            dist = 1.0 - ast_sym

            D[i, j] = dist
            D[j, i] = dist

    dist_mean, dist_max, dist_min, dist_std = get_stats_pairwise_distances(D)
    return float(dist_mean), float(dist_max), float(dist_min), float(dist_std),D

def compute_semantic_distance(texts, model=None, model_name=''):
    r = len(texts)
    '''
    if model_name in ['Qwen3-0.6B','Qwen3-4B','Qwen3-8B','nemotron']:
        instruct = 'Instruct:Given a code, retrieve semantically similar codes\nQuery:\n'
        query_texts = [instruct + s for s in texts]
        m_distance = model.similarity(model.encode(query_texts), model.encode(texts))
    else:
    '''
    m_distance = model.similarity(model.encode(texts), model.encode(texts))
    torch.cuda.empty_cache()

    for i in range(r):
        for j in range(r):
            m_distance[i][j] = 1 - m_distance[i][j]
            if i!=j and m_distance[i][j] < 0.000001:
                m_distance[i][j]= 0.000001
            elif i == j:
                m_distance[i][j]=0

    m_distance = np.round(np.asarray(m_distance),6)

    for i in range(r):
        for j in range(r):
            if i!=j and m_distance[i][j] != m_distance[j][i]:
                #print('Smoothing distance matrix',i,j, m_distance[i][j], m_distance[j][i])
                new_distance = (m_distance[i][j] + m_distance[j][i])/2
                m_distance[i][j] = new_distance
                m_distance[j][i] = new_distance

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


    #q10 = np.percentile(values, 10)
    #q90 = np.percentile(values, 90)

    #k = kurtosis(values, fisher=True, bias=False)

    return float(mean_dist), float(max_dist), float(min_dist), float(std_dist) #, float(q90-q10), float(k)