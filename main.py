import os
import copy
import json
import csv
import argparse
from tqdm import tqdm
import shutil
import subprocess
import re
import pickle

from sentence_transformers import SentenceTransformer

from generation.code_generation import generate, create_agents
from generation.utils import extract_all_comments

from evaluation.eval_codeguard import codeql_create_db, codeql_analyze, run_test_codeguard, update_stat_codeguard
from evaluation.eval_humaneval import update_stat_humaneval
from evaluation.eval_bigcodebench import update_stat_bigcodebench
from evaluation.utils import init_stat, update_stat_P_true

from analysis.report import report_codeguard, report

from agent.python_developer import python_developer
from agent.analyst_functionality import analyst_functionality
from agent.analyst_security import analyst_security
from agent.python_tester import python_tester
from agent.python_reviewer import python_reviewer
from agent.reviewer_functionality import reviewer_functionality
from agent.role_play import SYS_ANALYST_FUNCTIONALITY,SYS_ANALYST_SECURITY, PYTHON_DEVELOPER, SYS_REVIEWER_FUNCTIONALITY, SYS_ANALYST_PROBLEM
from agent.role_play import TEAM_FCF, TEAM_FSC, TEAM_FC, TEAM_SC, TEAM_CF, TEAM_CR



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='humaneval')
parser.add_argument('--lang', type=str, default='python')
parser.add_argument('--output_path', type=str, default='output.jsonl')


parser.add_argument('--do_generation', action='store_true')
parser.add_argument('--do_evaluation', action='store_true')
parser.add_argument('--do_report', action='store_true')

parser.add_argument('--signature', action='store_true')
parser.add_argument('--model', type=str, default='codellama/CodeLlama-7b-Instruct-hf')
parser.add_argument('--multi_agent', type=str, default='none')
parser.add_argument('--max_round', type=int, default=10)

parser.add_argument('--max_tokens', type=int, default=512) 
parser.add_argument('--majority', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--top_p', type=float, default=0.95)

parser.add_argument('--fail_list', type=list, default=[])
parser.add_argument('--append', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument("--timeout", type=float, default=10, help="how many seconds to wait during execution for each test case")
args = parser.parse_args()

'''
mistralai/Mistral-7B-Instruct-v0.2
codellama/CodeLlama-7b-Instruct-hf
deepseek-ai/deepseek-coder-7b-instruct-v1.5
Qwen/Qwen2.5-Coder-7B-Instruct
'''


def read_jsonl_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_object = json.loads(line.strip())
                data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")
    return data

if __name__ == '__main__':
    
    # load data
    if args.dataset=='codeguard_python.jsonl':
        with open(args.dataset, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif args.dataset=='HumanEval.jsonl' or args.dataset=='bigCodeBench_hard.jsonl':
        data = read_jsonl_file(args.dataset)
    

    base_dir = os.path.join('output', args.model.replace('/','_'), args.multi_agent, args.dataset.replace('.jsonl',''),)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if args.do_generation:
        agents = create_agents(args.multi_agent, args.model)
    elif args.do_evaluation or args.do_report:
        if args.dataset=='codeguard_python.jsonl':
            agents = create_agents('funcReviewer_secReviewer', args.model)
        else:
            agents = create_agents('funcReviewer', args.model)

    pbar = tqdm(total=len(data), desc="Processing")

    # for each code generation task targeting a vulnerability type and its scenario
    for task in data:

        pbar.update(1)
        
        if args.dataset=='codeguard_python.jsonl':
            output_dir = os.path.join(base_dir, task['cwe'].lower()+'_'+task['scenario'])
        elif args.dataset=='HumanEval.jsonl':
            output_dir = os.path.join(base_dir, task['task_id'].replace('/','_'))
        elif args.dataset=='bigCodeBench_hard.jsonl':
            output_dir = os.path.join(base_dir, task['task_id'])
            

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.do_generation:
            if args.dataset=='codeguard_python.jsonl':
                context = task['file_context'] + task['func_context']
                language = task['language']
            elif args.dataset=='HumanEval.jsonl':
                context = task['prompt']
                language = 'py'
            elif args.dataset=='bigCodeBench_hard.jsonl':
                context = task['code_prompt'] + '\n"""\n' + task['instruct_prompt_clean'] +'\n"""'
                language = 'py'
            #codes, plans, tests, reviews = generate(task['file_context'], task['func_context'], args.multi_agent)
            
            reports = generate(context, agents, args.multi_agent)

            for r_type, r in reports.items():
                if len(r) == 0:
                    continue

                report_dir = os.path.join(output_dir, r_type)
                if os.path.isdir(report_dir):  
                    shutil.rmtree(report_dir)
                os.makedirs(report_dir)

                if r_type == 'code' or r_type == 'test':
                    suffix = language
                else:
                    suffix = 'json'
                
                for i, content in enumerate(r):
                    fname = str(i)+'.'+suffix
                    file = os.path.join(report_dir,fname)
                    
                    with open(file, "w", encoding="utf-8") as f:
                        f.write(content)


        if args.do_evaluation:
            if args.dataset=='codeguard_python.jsonl':

                unittest_file = os.path.join('data','unit_test', task['cwe'].lower(), task['scenario'], 'functional.py')
                stat_file = os.path.join(output_dir, 'stat.json')
                codeql_file = os.path.join(output_dir, 'codeql.csv')
                codeql_db = os.path.join(output_dir, 'codeql_db')
                code_dir = os.path.join(output_dir, 'code')
                plan_dir = os.path.join(output_dir, 'func_plan')

                if os.path.isdir(output_dir): 

                    file_names = [s for s in os.listdir(code_dir) if s.endswith('py')] 

                    if os.path.exists(stat_file):
                        os.remove(stat_file)
                    init_stat(file_names, stat_file)

                    run_test_codeguard(task, code_dir, unittest_file, codeql_db, codeql_file)

                    update_stat_codeguard(file_names, codeql_file, stat_file, code_dir)
                    update_stat_P_true(file_names, stat_file, code_dir, extract_all_comments(task['func_context']),agents['reviewer_func'])
                    update_stat_P_true(file_names, stat_file, code_dir, '', agents['analyst_sec'], topic='self_sec')
            
            elif args.dataset=='HumanEval.jsonl':

                code_dir = os.path.join(output_dir, 'code')
                stat_file = os.path.join(output_dir, 'stat.json')
                
                if os.path.isdir(output_dir): 
                    file_names = [s for s in os.listdir(code_dir) if s.endswith('py')] 
                
                if os.path.exists(stat_file):
                    os.remove(stat_file)
                init_stat(file_names, stat_file)

                update_stat_humaneval(file_names, stat_file, task['test'], task['entry_point'], code_dir)
                
                update_stat_P_true(file_names, stat_file, code_dir, extract_all_comments(task['prompt']),agents['reviewer_func'])

            elif args.dataset=='bigCodeBench_hard.jsonl':

                code_dir = os.path.join(output_dir, 'code')
                stat_file = os.path.join(output_dir, 'stat.json')

                if os.path.isdir(output_dir): 
                    file_names = [s for s in os.listdir(code_dir) if s.endswith('py')] 

                if os.path.exists(stat_file):
                    os.remove(stat_file)
                init_stat(file_names, stat_file)
                print('------ Test BigCodeBench ',task['task_id'])
                update_stat_bigcodebench(file_names, stat_file, task['test'], task['entry_point'], code_dir)
                update_stat_P_true(file_names, stat_file, code_dir, task['instruct_prompt_clean'],agents['reviewer_func'])
                # update_stat_bigcodebench(file_names, stat_file, task['test'], task['entry_point'], code_dir,task['code_prompt'] + '\n' + task['canonical_solution'])


    pbar.close()

    if args.do_report:
        model = dict()
        model['allMiniLM'] = SentenceTransformer("all-MiniLM-L6-v2")
        model['Qwen3'] = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        model['modernbert'] = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
        model['equivalence'] = agents['reviewer_func']

        with open('threshold.pkl', 'rb') as file:
            thresholds = pickle.load(file)

        if args.dataset=='codeguard_python.jsonl':
            report_codeguard(base_dir,multi_agent=args.multi_agent, data_dir='data')
        else:
            report(base_dir, multi_agent=args.multi_agent, model=model, data=data, dataset=args.dataset, thresholds=thresholds)

        
                        