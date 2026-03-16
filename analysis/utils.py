import os
import json
import re
import math
import numpy as np
from numpy import mean
import copy
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from generation.utils import extract_all_comments, extract_before_fences

from scipy.sparse import csr_matrix
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