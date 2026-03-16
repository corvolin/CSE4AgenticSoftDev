import os
import subprocess
import json
import csv
import numpy as np
import ast
import statistics
#from bert_score import score

from evaluation.utils import compute_bleu, compute_contradiction_score, calculate_selfBleu, extract_terminal_pairs, analyze_json_structure


def codeql_create_db(out_src_dir, out_db_dir):
    cmd = '~/codeql/codeql database create {} --quiet --language=python --overwrite --source-root {}'
    cmd = cmd.format(out_db_dir, out_src_dir)
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,timeout=60)

def codeql_analyze(check_ql, out_db_dir, out_csv_path):
    cmd = '~/codeql/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}'
    cmd = cmd.format(out_db_dir, check_ql, out_csv_path, os.path.expanduser('~/.codeql/packages/codeql/'))
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,timeout=60)

def run_test_codeguard(task, code_dir, unittest_file, codeql_db, codeql_file):
    try:
        # functional correctness
        subprocess.call(['python', unittest_file, '--path', code_dir],timeout=30)
    except subprocess.TimeoutExpired:
        print('Unit test timeout')

    # codeql security
    try:
        codeql_create_db(code_dir, codeql_db)
        codeql_analyze(task['check_ql'], codeql_db, codeql_file)
        subprocess.run(f'rm -rf {codeql_db}', shell=True, stdout=subprocess.DEVNULL,timeout=30)
    except subprocess.TimeoutExpired:
        print('CodeQL timeout')

def update_stat_codeguard(all_fnames, csv_file, stat_file, code_dir):

    vuls = set()

    if os.path.exists(csv_file):
        with open(csv_file) as csv_f:
            reader = csv.reader(csv_f)
            for row in reader:
                if len(row) < 5: continue
                out_src_fname = row[-5].replace('/', '')
                vuls.add(out_src_fname)


    if os.path.exists(stat_file):
        with open(stat_file) as f:
            stat = json.load(f)
    for fname in all_fnames:
        if fname in vuls:
            stat[fname]['sec'] = False
        else:
            with open(os.path.join(code_dir,fname)) as code_file:
                code_str = code_file.read()
                try:
                    # Parse the code into an AST
                    tree = ast.parse(code_str)
                    stat[fname]['sec'] = True
                except SyntaxError as e:
                    stat[fname]['sec'] = False
            
    with open(stat_file, 'w') as f:
        json.dump(stat, f, indent=4)

def update_stat_plan_quality(all_fnames, base_plan, plan_dir, stat_file):
    if os.path.exists(stat_file):
        with open(stat_file) as f:
            stat = json.load(f)

    for fname in all_fnames:
        plan_file = os.path.join(plan_dir, fname.replace('py','json'))
        with open(plan_file) as f:
            plan = f.read()
            
        stat[fname]['plan_bert_score'] = f"{score([plan],[base_plan],lang='en')[2].item():.4f}"
        stat[fname]['plan_BLEU'] = f"{compute_bleu(plan,base_plan):.4f}"
        stat[fname]['plan_size'] = f"{len(plan):.2f}"
        stat[fname]['plan_self_BLEU'] = 0
        stat[fname]['plan_contradiction_score'] = 0
        stat[fname]['plan_max_depth'] = 0
        stat[fname]['plan_total_keys'] = 0
        stat[fname]['plan_keys_per_depth'] = {}
        stat[fname]['plan_average_value_length'] = 0

        try:
            with open(plan_file, 'r') as f:
                plan = json.load(f)
                stat[fname]['plan_contradiction_score'] = f"{compute_contradiction_score(plan,base_plan):.2f}"
                stat[fname]['plan_self_BLEU'] = calculate_selfBleu(extract_terminal_pairs(plan))
                analysis = analyze_json_structure(plan)
                stat[fname]['plan_max_depth'] = analysis["max_depth"]
                stat[fname]['plan_total_keys'] = analysis["total_keys"]
                stat[fname]['plan_keys_per_depth'] = analysis["keys_per_depth"]
                stat[fname]['plan_average_value_length'] = analysis["average_value_length"]
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}",f)
        except Exception as e:
            print(f"Error reading file: {e}")
        
    with open(stat_file, 'w') as f:
        json.dump(stat, f, indent=4)