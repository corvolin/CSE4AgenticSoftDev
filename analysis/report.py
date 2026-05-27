import os
import csv
import statistics
import json
import time

import numpy as np
from tqdm import tqdm
from agent.reviewer_functionality import reviewer_functionality
from agent.role_play import REVIEWER_FUNC_EQ
from generation.utils import extract_all_comments
from evaluation.utils import compute_pass_at_k, self_eval_per_code
from analysis.code_semantic_entropy import compute_semantic_entropy_clustering, compute_semantic_entropy_equivalence
from analysis.code_semantic_entropy import compute_lexical_entropy_equivalence, compute_structural_entropy_equivalence 
from analysis.code_semantic_entropy import compute_semantic_entropy_HDBSCAN, compute_semantic_entropy_DBSCAN, compute_semantic_entropy_threshold
from analysis.code_semantic_entropy import compute_mst_pruning_entropy, compute_joint_cluter_entropy
from analysis.code_semantic_entropy import compute_mst_DBSCAN_entropy, compute_mean_threshold_DBSCAN_entropy, compute_mst_mean_threshold_DBSCAN_entropy
from analysis.code_semantic_entropy import compute_ratio_DBSCAN_entropy, compute_mst_pruning_ratio_DBSCAN_entropy
from analysis.code_semantic_distance import compute_semantic_distance, compute_lexical_distance, compute_structural_distance
from analysis.code_semantic_alignment import compute_semantic_alignment
from analysis.code_semantic_MST import compute_semantic_MST
from analysis.utils import get_list_code_threshold, get_list_func_plan_threshold, get_thershold_by_reference, get_column, majority_true, edge_type_stats

def report_codeguard(base_dir, multi_agent='none', data_dir=''):
    csv_file = os.path.join(base_dir,'report.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)
    with open(csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)  # Create a writer object :contentReference[oaicite:1]{index=1}
        if multi_agent=='analyst_coder':
            writer.writerow(['vulnerability','pass_at_1','secure_at_1','pass_secure_at_1','bertscore','BLEU','contradiction_score','plan_size','max_depth','total_keys','average_value_length'])
        else:
            writer.writerow(['vulnerability','pass_at_1','secure_at_1','pass_secure_at_1'])
   
        l_pass_at_1 = []
        l_secure_at_1 = []
        l_pass_secure_at_1 = []
        l_bertscore = []
        l_BLEU = []
        l_contradiction_score = []
        l_max_depth = []
        l_total_keys = []
        l_average_value_length = []
        l_plan_size = []
    
        for vulnerability in os.listdir(base_dir):
            n_correct = 0
            n_secure = 0
            n_correct_secure = 0
            n_total = 0

            vul_dir = os.path.join(base_dir, vulnerability)
            if 'report' in vul_dir:
                continue
            for senario in os.listdir(vul_dir):
                l_module_bertscore = []
                l_module_bleu = []
                l_module_contradiction_score = []
                l_module_max_depth = []
                l_module_total_keys = []
                l_module_average_value_length = []
                l_module_plan_size = []

                stat_file = os.path.join(vul_dir,senario,'stat.json')
                plan_dir = os.path.join(vul_dir,senario,'plan')

                with open(stat_file, 'r') as f:
                    stat = json.load(f)

                for module in stat:
                    
                    n_total = n_total+1
                    if stat[module]['functional'] and stat[module]['sec']:
                        n_correct = n_correct+1
                        n_secure = n_secure+1
                        n_correct_secure = n_correct_secure+1
                    elif stat[module]['functional']:
                        n_correct = n_correct+1
                    elif stat[module]['sec']:
                        n_secure = n_secure+1

                    if multi_agent=='analyst_coder':
                        l_module_bertscore.append(float(stat[module]['plan_bert_score']))
                        l_module_bleu.append(float(stat[module]['plan_BLEU']))
                        l_module_contradiction_score.append(float(stat[module]['plan_contradiction_score']))
                        l_module_plan_size.append(float(stat[module]['plan_size']))
                        l_module_max_depth.append(float(stat[module]['plan_max_depth']))
                        l_module_total_keys.append(float(stat[module]['plan_total_keys']))
                        l_module_average_value_length.append(float(stat[module]['plan_average_value_length']))

                pass_at_1 = compute_pass_at_k(n_total,n_correct,1)
                secure_at_1 = compute_pass_at_k(n_total,n_secure,1)
                pass_secure_at_1 = compute_pass_at_k(n_total,n_correct_secure,1)


                l_pass_at_1.append(pass_at_1)
                l_secure_at_1.append(secure_at_1)
                l_pass_secure_at_1.append(pass_secure_at_1)


                if multi_agent=='analyst_coder':
                    l_bertscore.append(statistics.mean(l_module_bertscore))
                    l_BLEU.append(statistics.mean(l_module_bleu))
                    l_contradiction_score.append(statistics.mean(l_module_contradiction_score))
                    l_plan_size.append(statistics.mean(l_module_plan_size))
                    l_max_depth.append(statistics.mean(l_module_max_depth))
                    l_total_keys.append(statistics.mean(l_module_total_keys))
                    l_average_value_length.append(statistics.mean(l_module_average_value_length))

                

            if multi_agent=='analyst_coder':
                writer.writerow([vulnerability, pass_at_1, secure_at_1, pass_secure_at_1,f"{statistics.mean(l_module_bertscore):.2f}",f"{statistics.mean(l_module_bleu):.2f}",f"{statistics.mean(l_module_contradiction_score):.2f}",f"{statistics.mean(l_module_plan_size):.2f}",f"{statistics.mean(l_module_max_depth):.2f}",f"{statistics.mean(l_module_total_keys):.2f}",f"{statistics.mean(l_module_average_value_length):.2f}"])
            else:
                writer.writerow([vulnerability, pass_at_1, secure_at_1, pass_secure_at_1])
            
        if multi_agent=='analyst_coder':
            writer.writerow(['average', f"{statistics.mean(l_pass_at_1):.2f}", f"{statistics.mean(l_secure_at_1):.2f}", f"{statistics.mean(l_pass_secure_at_1):.2f}",f"{statistics.mean(l_bertscore):.2f}", f"{statistics.mean(l_BLEU):.2f}", f"{statistics.mean(l_contradiction_score):.2f}",f"{statistics.mean(l_plan_size):.2f}",f"{statistics.mean(l_max_depth):.2f}",f"{statistics.mean(l_total_keys):.2f}",f"{statistics.mean(l_average_value_length):.2f}"])

        else:
            writer.writerow(['average', f"{statistics.mean(l_pass_at_1):.2f}", f"{statistics.mean(l_secure_at_1):.2f}", f"{statistics.mean(l_pass_secure_at_1):.2f}"])

def report(base_dir, multi_agent='none', model=None, data=None, dataset='', thresholds=None):
    csv_file = os.path.join(base_dir,'report.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)

    l_models = list(model.keys())
    if 'equivalence' in l_models:
        l_models.remove('equivalence')

    write_row_name = False
    do_self_eval = False

    """
    for m in model.keys():
        if m == 'equivalence':
            continue
        row.append('code_se_logit_'+m)
        row.append('code_se_logit_inverse_'+m)

    if 'funcAnalyst' in multi_agent:
        for m in model.keys():
            if m == 'equivalence':
                continue
            row.append('func_plan_se_logit_'+m)
            row.append('func_plan_se_logit_inverse_'+m)
    """
    
    with open(csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile) 

        pbar = tqdm(total=len(os.listdir(base_dir)), desc="Processing")

        for task in os.listdir(base_dir):
            pbar.update(1)
            print('-------- report', task, '---------')

            requirement = ''
            row_value = dict()
            row_value['task_id'] = task
            for d in data:
                if d['task_id'] == task:
                    if dataset == 'HumanEval.jsonl':
                        requirement = extract_all_comments(d['prompt'])
                    elif dataset == 'bigCodeBench_hard.jsonl':
                        requirement = d['instruct_prompt_clean']
                    # print(requirement)
                    break
                    
            n_correct = 0
            n_p_true = 0
            n_voting = 0
            n_total = 0
            l_func_plan = []
            l_code = []
            l_func_review = []

            task_dir = os.path.join(base_dir, task)
            if 'report' in task_dir:
                continue
            
            stat_file = os.path.join(task_dir,'stat.json')
            with open(stat_file, 'r') as f:
                stat = json.load(f)

            l_labels = []
            for module in stat:
                
                n_total = n_total+1
                if stat[module]['functional']:
                    n_correct = n_correct+1
                    l_labels.append(True)
                else:
                    l_labels.append(False)

                if stat[module]['self_true']:
                    n_p_true = n_p_true+1
                if majority_true(stat[module]['voting_true']):
                    n_voting = n_voting+1
                    
                with open(os.path.join(task_dir,'code', module), 'r') as f:
                    code = f.read()
                    l_code.append(code)

                if 'funcAnalyst' in multi_agent:
                    with open(os.path.join(task_dir,'func_plan', module.replace('.py','.json')), 'r') as f:
                        func_plan = f.read()
                        l_func_plan.append(func_plan)

                if 'funcReviewer' in multi_agent:
                    with open(os.path.join(task_dir,'func_review', module.replace('.py','.json')), 'r') as f:
                        func_review = f.read()
                        l_func_review.append(func_review)


            """
            n_p_true = 0
            n_p_true_time = 0

            if do_self_eval:
                start = time.perf_counter() 
                for c in l_code:
                    if self_eval_per_code(c, model['equivalence'], requirement, 'self_true'):
                        n_p_true = n_p_true + 1
                #print(n_p_true)
                end = time.perf_counter()
                n_p_true_time = end - start

            code_se_equivalence = 0
            code_se_equivalence_time = 0

            start = time.perf_counter() 
            for code in l_code:
                reviewer_func_res = model['equivalence'].task_review(code)
                model['equivalence'].history_message_clear()
            end = time.perf_counter()
            func_review_agent_time = end - start 
            for m in model.keys():
                
            
            if 'funcAnalyst' in multi_agent:

                for m in model.keys():
                    if m == 'equivalence':
                        continue
                    else:
                        #for threshold_func in l_threshold_func:
                            # threshold = get_thershold_by_reference(thresholds[m]['func_review_align'], d_func_align[m], thresholds[m]['func_plan_dist'], threshold_func)  
                        plan_se = compute_semantic_entropy_clustering(l_func_plan, model[m], None, thresholds[m]['func_review_align'], d_func_align[m], thresholds[m]['func_plan_dist'])
                        # plan_se = compute_semantic_entropy_clustering(l_func_plan, model[m], 0.005)
                        l_plan_se.extend(plan_se)
                            
                        # l_func_plan_se['func_plan_se_'+m].append(func_plan_se)
                    #l_plan_se.extend(compute_semantic_entropy_clustering(l_func_plan, model[m], get_list_func_plan_threshold(m,thresholds)))

            """


            row_value['pass_at_1'] = compute_pass_at_k(n_total,n_correct,1)
            row_value['majority_voting_true'] = n_voting
            
            #l_code_se.append(compute_semantic_entropy_identical(l_code))
            if 'funcReviewer' in multi_agent:
                
                #func_review_dist_matrix = report_dist(l_func_review, model, row_value, requirement, 'func_review')

                report_align(l_func_review, model, row_value, requirement, 'func_review')
                # report_MST(l_func_review, model, row_value, requirement, 'func_review', func_review_dist_matrix)
                #report_HDBSCAN(l_func_review, model, row_value, requirement, 'func_review', func_review_dist_matrix)
                #report_lexical_equivalence(l_func_review, row_value, 'func_review')
                #report_edge(func_review_dist_matrix, l_labels, row_value, 'func_review')
                # report_mst_pruning_entropy(func_review_dist_matrix, row_value, 'func_review')
            
            
            if 'coder' in multi_agent:
                
                code_dist_matrix = report_dist(l_code, model, row_value, requirement, 'code')

                #report_align(l_code, model, row_value, requirement, 'code')
                # report_MST(l_code, model, row_value, requirement, 'code', code_dist_matrix)
                report_HDBSCAN(l_code, model, row_value, requirement, 'code', code_dist_matrix)
                report_DBSCAN(l_code, model, row_value, requirement, 'code', code_dist_matrix)
                #report_scaled(l_code, model, row_value, thresholds, 'func_review_align', 'func_review_align_mean','code_dist', requirement, 'code', code_dist_matrix)
                #lexical_labels = report_lexical_equivalence(l_code, row_value, 'code')
                structural_labels = report_structural_equivalence(l_code, row_value, 'code')
                report_edge(code_dist_matrix, l_labels, row_value, 'code')
                #report_mst_pruning_entropy(code_dist_matrix, row_value, 'code', structural_labels)
                report_mean_threshold_DBSCAN_entropy(code_dist_matrix, row_value, 'code', structural_labels)
                report_mst_ratio_threshold_entropy(code_dist_matrix, row_value, 'code', structural_labels)
            '''
            if 'funcAnalyst' in multi_agent:
                
                func_plan_dist_matrix = report_dist(l_func_plan, model, row_value, requirement, 'func_plan')

                report_align(l_func_plan, model, row_value, requirement, 'func_plan')
                # report_MST(l_func_plan, model, row_value, requirement, 'func_plan', func_plan_dist_matrix)
                report_HDBSCAN(l_func_plan, model, row_value, requirement, 'func_plan', func_plan_dist_matrix)
                report_scaled(l_func_plan, model, row_value, thresholds, 'func_review_align', 'func_review_align_mean','func_plan_dist', requirement, 'func_plan', func_plan_dist_matrix)
                report_lexical_equivalence(l_func_plan, row_value, 'func_plan')
                report_edge(func_plan_dist_matrix, l_labels, row_value, 'func_plan')
                # report_mst_pruning_entropy(func_plan_dist_matrix, row_value, 'func_plan')
            '''
            if not write_row_name:
                writer.writerow(list(row_value.keys()))
                write_row_name = True
                print(list(row_value.keys()))
            writer.writerow(list(row_value.values()))

        pbar.close()

# compute the semantic entropy of a set of texts by class identical texts together
def report_lexical_equivalence(texts, row_value, prefix=''):
    entropy, labels = compute_lexical_entropy_equivalence(texts)
    row_value[prefix+'_lexical_entropy_equivalence'] = entropy
    return labels

# compute the semantic entropy of a set of texts by class identical structures together
def report_structural_equivalence(texts, row_value, prefix=''):
    entropy, labels =compute_structural_entropy_equivalence(texts)
    row_value[prefix+'_structural_entropy_equivalence'] = entropy
    return labels

# compute cosine similarity between given requirement and a set of texts        
def report_align(texts, model, row_value, requirement='', prefix=''):
    for m in model.keys():
        if m == 'equivalence':
            continue
        else:
            # semantic alignment
            start = time.perf_counter() 
            align_mean, align_max, align_min, align_std = compute_semantic_alignment(texts, requirement, model[m])
            end = time.perf_counter()
            align_time = end - start 

            row_value[prefix+'_align_mean_'+m] = align_mean
            row_value[prefix+'_align_max_'+m] = align_max
            row_value[prefix+'_align_min_'+m] = align_min
            row_value[prefix+'_align_std_'+m] = align_std
            row_value[prefix+'_align_time_'+m] = align_time

# compute pair-wise cosine similarity between a set of texts
def report_dist(texts, model, row_value, requirement='', prefix=''):
    d_dist_matrix = dict()
    for m in model.keys():
        if m == 'equivalence':
            continue
        else:
            start = time.perf_counter() 
            dist_mean, dist_max, dist_min, dist_std, dist_matrix = compute_semantic_distance(texts, model[m], m)
            end = time.perf_counter()
            dist_time = end - start 

            row_value[prefix+'_dist_mean_'+m] = dist_mean
            row_value[prefix+'_dist_max_'+m] = dist_max
            row_value[prefix+'_dist_min_'+m] = dist_min
            row_value[prefix+'_dist_std_'+m] = dist_std
            row_value[prefix+'_dist_time_'+m] = dist_time

            d_dist_matrix[m] = dist_matrix

    m = 'lexical'
    start = time.perf_counter() 
    dist_mean, dist_max, dist_min, dist_std, dist_matrix = compute_lexical_distance(texts)
    end = time.perf_counter()
    dist_time = end - start 

    row_value[prefix+'_dist_mean_'+m] = dist_mean
    row_value[prefix+'_dist_max_'+m] = dist_max
    row_value[prefix+'_dist_min_'+m] = dist_min
    row_value[prefix+'_dist_std_'+m] = dist_std
    row_value[prefix+'_dist_time_'+m] = dist_time

    d_dist_matrix[m] = dist_matrix

    if prefix == 'code':
        m = 'structural'
        start = time.perf_counter() 
        dist_mean, dist_max, dist_min, dist_std, dist_matrix = compute_structural_distance(texts)
        end = time.perf_counter()
        dist_time = end - start 

        row_value[prefix+'_dist_mean_'+m] = dist_mean
        row_value[prefix+'_dist_max_'+m] = dist_max
        row_value[prefix+'_dist_min_'+m] = dist_min
        row_value[prefix+'_dist_std_'+m] = dist_std
        row_value[prefix+'_dist_time_'+m] = dist_time

        d_dist_matrix[m] = dist_matrix
    
    return d_dist_matrix
# compute the minimum spanning tree of the pair-wise cosine similarity of a set of texts
# if distance matrix is provided, do not re-compute 
def report_MST(texts, model, row_value, requirement='', prefix='', dist_matrix=None):
    l_MST_time = []
    for m in model.keys():
        if m == 'equivalence':
            continue
        else:
            start = time.perf_counter() 
            MST_weight_sum , MST_weight_mean , MST_weight_max , MST_weight_min, MST_weight_std, MST_degree_mean, MST_degree_max, MST_leaf_count, MST_diameter_weight, MST_diameter_count = compute_semantic_MST(texts, model[m], distance_matrix=dist_matrix[m])
            end = time.perf_counter()
            l_MST_time.append(end - start)

            row_value[prefix+'_MST_weight_sum_'+m] = MST_weight_sum
            row_value[prefix+'_MST_weight_mean_'+m] = MST_weight_mean
            row_value[prefix+'_MST_weight_max_'+m] = MST_weight_max
            row_value[prefix+'_MST_weight_min_'+m] = MST_weight_min
            row_value[prefix+'_MST_weight_std_'+m] = MST_weight_std
            row_value[prefix+'_MST_degree_mean_'+m] = MST_degree_mean
            row_value[prefix+'_MST_degree_max_'+m] = MST_degree_max
            row_value[prefix+'_MST_leaf_count_'+m] = MST_leaf_count
            row_value[prefix+'_MST_diameter_weight_'+m] = MST_diameter_weight
            row_value[prefix+'_MST_diameter_count_'+m] = MST_diameter_count
            
    
    row_value[prefix+'_MST_time'] = np.mean(l_MST_time)

# compute the semantic entropy of a set of texts using HDBSCAN for class label
def report_HDBSCAN(texts, model, row_value, requirement='', prefix='', dist_matrix=None):
    l_se_HDBSCAN_time = []
    for m in model.keys():
        if m == 'equivalence':
            continue
        else:
            l_min_cluster_size = [2,3,4,5,6,7,8,9]
            start = time.perf_counter() 

            for mcs in l_min_cluster_size:
                se_HDBSCAN = compute_semantic_entropy_HDBSCAN(texts, model[m], distance_matrix=dist_matrix[m], min_cluster_size=mcs)
                row_value[prefix+'_se_HDBSCAN_'+m+'_'+str(mcs)] = se_HDBSCAN

            end = time.perf_counter()

            l_se_HDBSCAN_time.append((end - start)/len(l_min_cluster_size))
    
    row_value[prefix+'_se_HDBSCAN_time'] = np.mean(l_se_HDBSCAN_time)


# compute the semantic entropy of a set of texts using HDBSCAN for class label
def report_DBSCAN(texts, model, row_value, requirement='', prefix='', dist_matrix=None):
    l_se_DBSCAN_time = []
    for m in model.keys():
        if m == 'equivalence':
            continue
        else:
            l_epsilon = [round(x, 3) for x in np.linspace(0.01, 0.3, 14).tolist()]
            start = time.perf_counter() 

            for e in l_epsilon:
                se_HDBSCAN = compute_semantic_entropy_DBSCAN(texts, model[m], distance_matrix=dist_matrix[m], epsilon=e)
                row_value[prefix+'_se_DBSCAN_'+m+'_'+str(e)] = se_HDBSCAN

            end = time.perf_counter()

            l_se_DBSCAN_time.append((end - start)/len(l_epsilon))
    
    row_value[prefix+'_se_DBSCAN_time'] = np.mean(l_se_DBSCAN_time)

# compute the semantic entropy of a set of texts with a provided threshold value and function
# the value is used as the epsilon of core node's range for DBSCAN
def report_scaled(texts, model, row_value, thresholds, base_key, target_key, reference_key, requirement='', prefix='', dist_matrix=None):
    l_se_scaled_time = []
    for m in model.keys():
        if m == 'equivalence':
            continue
        else:
            threshold_function = ['linear', 'log', 'square', 'sigmoid', 'logit', 'linear_inverse', 'log_inverse', 'square_inverse', 'sigmoid_inverse', 'logit_inverse']
            for tf in threshold_function:
                start = time.perf_counter() 
                threshold = get_thershold_by_reference(thresholds[m][base_key], row_value[target_key+'_'+m], thresholds[m][reference_key], tf)
                se_scaled = compute_semantic_entropy_threshold(texts, threshold, model[m], distance_matrix=dist_matrix[m])
                end = time.perf_counter()

                row_value[prefix+'_se_'+tf+'_'+m] = se_scaled
                
                l_se_scaled_time.append(end - start)
                
    row_value[prefix+'_se_scaled_time'] = np.mean(l_se_scaled_time)

def report_edge(d_distance_matrix, labels, row_value, prefix=''):
    for m in d_distance_matrix.keys():
        if m == 'structural' and not prefix=='code':
            continue
        else:
            res = edge_type_stats(d_distance_matrix[m],labels)
            row_value[prefix+'_mst_time_'+m]=res['mst_time']
            res_type = ['global', 'mst']
            #edge_type = ['pass_pass', 'fail_fail', 'pass_fail', 'all']
            edge_type = ['all']
            value_type = ['mean', 'std', 'count', 'quantile', 'kurtosis', 'skew', 'peak_count', 'max_mode']
            for r in res_type:
                for e in edge_type:
                    for v in value_type:
                        row_value[prefix+'_'+e+'_dist_'+r+'_'+m+'_'+v] = res[r][e][v]

def report_mst_pruning_entropy( d_distance_matrix, row_value, prefix='', reference_labels=[]):
    criteria = ['mean', 'std', 'both']
    threshold = [ 0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999]
    for m in d_distance_matrix.keys():
        for c in criteria:
            for t in threshold:
                entropy, labels = compute_mst_pruning_entropy(d_distance_matrix[m], c,t)
                row_value[prefix+'_mst_pruning_entropy_'+c+'_'+str(t)+'_'+m] = entropy
                if m != 'structural' and m!= 'lexical':
                    row_value[prefix+'_union_mst_pruning_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'union')
                    row_value[prefix+'_intersection_mst_pruning_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'intersection')

                entropy, labels = compute_mst_DBSCAN_entropy(d_distance_matrix[m], c, t)
                row_value[prefix+'_mst_pruning_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = entropy
                if m != 'structural' and m!= 'lexical':
                    row_value[prefix+'_union_mst_pruning_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'union')
                    row_value[prefix+'_intersection_mst_pruning_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'intersection')


def report_mean_threshold_DBSCAN_entropy(d_distance_matrix, row_value, prefix='', reference_labels=[]):
    criteria = ['mean']
    threshold = [ 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    for m in d_distance_matrix.keys():
        for c in criteria:
            for t in threshold:
                entropy, labels = compute_mean_threshold_DBSCAN_entropy(d_distance_matrix[m], t)
                row_value[prefix+'_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = entropy
                if m != 'structural' and m!= 'lexical':
                    row_value[prefix+'_union_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'union')
                    row_value[prefix+'_intersection_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'intersection')

                entropy, labels = compute_mst_mean_threshold_DBSCAN_entropy(d_distance_matrix[m], t)
                row_value[prefix+'_DBSCAN_entropy_mst_'+c+'_'+str(t)+'_'+m] = entropy
                if m != 'structural' and m!= 'lexical':
                    row_value[prefix+'_union_DBSCAN_entropy_mst_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'union')
                    row_value[prefix+'_intersection_DBSCAN_entropy_mst_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'intersection')

def report_mst_ratio_threshold_entropy(d_distance_matrix, row_value, prefix='', reference_labels=[]):
    criteria = ['mean', 'median', 'min_mode', 'max_mode']
    threshold =  [ 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]

    for m in d_distance_matrix.keys():
        for c in criteria:
            for t in threshold:
                entropy, labels = compute_ratio_DBSCAN_entropy(d_distance_matrix[m], c, t)
                row_value[prefix+'_mst_ratio_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = entropy
                if m != 'structural' and m!= 'lexical':
                    row_value[prefix+'_union_mst_ratio_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'union')
                    row_value[prefix+'_intersection_mst_ratio_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'intersection')
                
                entropy, labels = compute_ratio_DBSCAN_entropy(d_distance_matrix[m], c, t, use_mst=False)
                row_value[prefix+'_global_ratio_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = entropy
                if m != 'structural' and m!= 'lexical':
                    row_value[prefix+'_union_global_ratio_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'union')
                    row_value[prefix+'_intersection_global_ratio_DBSCAN_entropy_'+c+'_'+str(t)+'_'+m] = compute_joint_cluter_entropy(labels, reference_labels, 'intersection')
               



