from agent.python_developer import python_developer
from agent.analyst_functionality import analyst_functionality
from agent.analyst_security import analyst_security
from agent.python_tester import python_tester
from agent.python_reviewer import python_reviewer
from agent.reviewer_functionality import reviewer_functionality

from agent.role_play import SYS_ANALYST_FUNCTIONALITY,SYS_ANALYST_SECURITY, PYTHON_DEVELOPER, SYS_REVIEWER_FUNCTIONALITY, SYS_ANALYST_PROBLEM
from agent.role_play import TEAM_FCF, TEAM_FSC, TEAM_FC, TEAM_SC, TEAM_CF, TEAM_CR

from generation.utils import remove_comments, extract_all_comments, extract_between_fences, extract_before_fences


def create_agents(multi_agent, model):

    agents = dict()
    
    if multi_agent=='voting':
        selected_models = ['mistralai/Mistral-7B-Instruct-v0.2','codellama/CodeLlama-7b-Instruct-hf','deepseek-ai/deepseek-coder-7b-instruct-v1.5','Qwen/Qwen2.5-Coder-7B-Instruct']
        #selected_models = ['codellama/CodeLlama-7b-Instruct-hf' ]
        for n in selected_models:
            if n!= model:
                agents[n] = reviewer_functionality(model=model, role=SYS_REVIEWER_FUNCTIONALITY)
    else:
        if 'coder' in multi_agent:
            agents['coder']=python_developer(model=model, role=PYTHON_DEVELOPER)
        if 'funcReviewer' in multi_agent:
            agents['reviewer_func'] = reviewer_functionality(model=model, role=SYS_REVIEWER_FUNCTIONALITY)
        if 'funcAnalyst' in multi_agent:
            agents['analyst_func'] = analyst_functionality(TEAM_FC,SYS_ANALYST_FUNCTIONALITY, '', '',model)
        if 'secAnalyst' in multi_agent or 'secReviewer' in multi_agent:
            agents['analyst_sec']= analyst_security(TEAM_SC, SYS_ANALYST_FUNCTIONALITY, '', '', model)
        
    return agents
    

def generate(context, agents, multi_agent='none', num_output=10, validate_limit=2):
    
    code_unfinished = remove_comments(context)
    requirement = extract_all_comments(context)

    reports = dict({
        'code': [],
        'func_plan': [],
        'sec_plan': [],
        'func_review': [],
        'sec_review': [],
        'task_estimate': []
    })

    if multi_agent == 'none':
        coder = python_developer('','','')

        for i in range(num_output):
            code = coder.task_coding(context)
            reports['code'].append(code)

    if multi_agent == 'coder':
        # Generate code
        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)
        #coder_chat = coder.task_coding(code_unfinished,topic='code',do_sample=False,num_beams=1,temperature=1.0,top_p=1.0)

        coder.history_message_clear()

        for i in range(num_output):
            coder_chat = coder.task_coding(code_unfinished,topic='code')
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content'])

            # Validate code
            #code = validate_code(code, file_context, coder, validate_limit)
            
            coder.history_message_clear()

            reports['code'].append(code)

    elif multi_agent == 'funcAnalyst_coder':

        analyst = agents['analyst_func']
        analyst.update_requirement_code(requirement, code_unfinished)
        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)

        analyst.history_message_clear()
        coder.history_message_clear()
        
        for i in range(num_output):
            # Generate Plan
            analyst_res = analyst.task_requirement(context)
            #plan = extract_between_fences(analyst_chat[-1]['generated_text'][-1]['content'])
            plan = extract_before_fences(analyst_res).strip()
            print('----------')
            print(plan)
            #plan = validate_plan(plan, analyst)

            # Generate code
            coder_chat = coder.task_coding(plan,topic='plan',do_sample=False,num_beams=1,temperature=1.0,top_p=1.0)
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content']).strip()

            # Validate code
            #code = validate_code(code, file_context, coder, validate_limit)
            analyst.history_message_clear()
            coder.history_message_clear()

            reports['code'].append(code)
            reports['func_plan'].append(plan)

    elif multi_agent == 'secAnalyst_coder':

        analyst = agents['analyst_sec']
        analyst.update_requirement_code(requirement, code_unfinished)
        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)
        
        analyst.history_message_clear()
        coder.history_message_clear()

        for i in range(num_output):
            # Generate security risk
            analyst_res = analyst.task_potential_vulnerability(code_unfinished, requirement)
            #plan = extract_between_fences(analyst_chat[-1]['generated_text'][-1]['content'])
            risk = extract_before_fences(analyst_res).strip()
            print('----------')
            print(risk)
            #plan = validate_plan(plan, analyst)

            # Generate code
            coder_chat = coder.task_coding(risk,topic='secure',do_sample=False,num_beams=1,temperature=1.0,top_p=1.0)
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content']).strip()

            # Validate code
            #code = validate_code(code, file_context, coder, validate_limit)
            analyst.history_message_clear()
            coder.history_message_clear()

            reports['code'].append(code)
            reports['sec_plan'].append(risk)

    elif multi_agent == 'funcAnalyst_secAnalyst_coder':
        
        analyst_func = agents['analyst_func']
        analyst_func.update_requirement_code(requirement, code_unfinished)
        analyst_sec = agents['analyst_sec']
        analyst_sec.update_requirement_code(requirement, code_unfinished)
        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)

        analyst_func.history_message_clear()
        analyst_sec.history_message_clear()
        coder.history_message_clear()

        for i in range(num_output):
            # Generate security risk
            analyst_func_res = analyst_func.task_requirement(context)
            analyst_sec_res = analyst_sec.task_potential_vulnerability(code_unfinished, requirement)
            
            plan = extract_before_fences(analyst_func_res).strip()
            risk = extract_before_fences(analyst_sec_res).strip()
            print('----------')
            print(plan)
            print(risk)
            #plan = validate_plan(plan, analyst)

            # Generate code
            coder_chat = coder.task_coding(plan+'\nBe aware of the following potential security risk and use mitigation when needed:\n'+risk,topic='plan',do_sample=False,num_beams=1,temperature=1.0,top_p=1.0)
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content']).strip()

            # Validate code
            #code = validate_code(code, file_context, coder, validate_limit)

            analyst_func.history_message_clear()
            analyst_sec.history_message_clear()
            coder.history_message_clear()

            reports['code'].append(code)
            reports['func_plan'].append(plan)
            reports['sec_plan'].append(risk)

        
    elif multi_agent == 'coder_funcReviewer':

        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)
        reviewer_func = agents['reviewer_func']
        reviewer_func.update_requirement_code(requirement, code_unfinished)

        coder.history_message_clear()
        reviewer_func.history_message_clear()

        for i in range(num_output):
            

            # Generate code
            coder_chat = coder.task_coding(code_unfinished,topic='code')
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content']).strip()

            reviewer_func_res = reviewer_func.task_review(code)
            review = extract_before_fences(reviewer_func_res).strip()
            review = extract_before_fences(review,fence='\n\n').strip()
            print('----------')
            print(review)

            coder.history_message_clear()
            reviewer_func.history_message_clear()

            reports['code'].append(code)
            reports['func_review'].append(review)

    elif multi_agent == 'coder_funcReviewer_secReviewer':

        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)
        reviewer_func = agents['reviewer_func']
        reviewer_func.update_requirement_code(requirement, code_unfinished)
        analyst_sec = agents['analyst_sec']
        analyst_sec.update_requirement_code(requirement, code_unfinished)
    
        coder.history_message_clear()
        reviewer_func.history_message_clear()
        analyst_sec.history_message_clear()

        for i in range(num_output):
            

            # Generate code
            coder_chat = coder.task_coding(code_unfinished,topic='code')
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content']).strip()

            reviewer_func_res = reviewer_func.task_review(code)
            review = extract_before_fences(reviewer_func_res).strip()
            review = extract_before_fences(review,fence='\n\n').strip()
            print('----------')
            print(review)
            analyst_sec_res = analyst_sec.task_potential_vulnerability(code, review)
            new_risk = extract_before_fences(analyst_sec_res).strip()

            coder.history_message_clear()
            reviewer_func.history_message_clear()
            analyst_sec.history_message_clear()

            reports['code'].append(code)
            reports['func_review'].append(review)
            reports['sec_review'].append(new_risk)
        
    elif multi_agent == 'funcAnalyst_coder_funcReviewer':

        analyst_func = agents['analyst_func']
        analyst_func.update_requirement_code(requirement, code_unfinished)
        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)
        reviewer_func = agents['reviewer_func']
        reviewer_func.update_requirement_code(requirement, code_unfinished)

        analyst_func.history_message_clear()
        coder.history_message_clear()
        reviewer_func.history_message_clear()

        for i in range(num_output):
            # Generate security risk
            analyst_func_res = analyst_func.task_requirement(context)
            
            #print(analyst_func_res)
            plan = extract_before_fences(analyst_func_res).strip()
            print('----------plan')
            print(plan)
            #plan = validate_plan(plan, analyst)

            # Generate code
            coder_chat = coder.task_coding(plan,topic='plan',do_sample=False,num_beams=1,temperature=1.0,top_p=1.0)
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content']).strip()

            reviewer_func_res = reviewer_func.task_review(code)
            review = extract_before_fences(reviewer_func_res).strip()
            review = extract_before_fences(review,fence='\n\n').strip()
            print('----------review')
            print(review)

            analyst_func.history_message_clear()
            coder.history_message_clear()
            reviewer_func.history_message_clear()

            reports['code'].append(code)
            reports['func_plan'].append(plan)
            reports['func_review'].append(review)

    elif multi_agent == 'funcAnalyst_coder_funcReviewer_secReviewer':

        analyst_func = agents['analyst_func']
        analyst_func.update_requirement_code(requirement, code_unfinished)
        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)
        reviewer_func = agents['reviewer_func']
        reviewer_func.update_requirement_code(requirement, code_unfinished)
        analyst_sec = agents['analyst_sec']
        analyst_sec.update_requirement_code(requirement, code_unfinished)

        analyst_func.history_message_clear()
        coder.history_message_clear()
        reviewer_func.history_message_clear()
        analyst_sec.history_message_clear()


        for i in range(num_output):
            # Generate security risk
            analyst_func_res = analyst_func.task_requirement(context)
            
            plan = extract_before_fences(analyst_func_res).strip()
            print('----------')
            print(plan)
            #plan = validate_plan(plan, analyst)

            # Generate code
            coder_chat = coder.task_coding(plan,topic='plan',do_sample=False,num_beams=1,temperature=1.0,top_p=1.0)
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content']).strip()

            reviewer_func_res = reviewer_func.task_review(code)
            review = extract_before_fences(reviewer_func_res).strip()
            review = extract_before_fences(review,fence='\n\n').strip()
            print('----------')
            print(review)
            analyst_sec_res = analyst_sec.task_potential_vulnerability(code, review)
            new_risk = extract_before_fences(analyst_sec_res).strip()

            analyst_func.history_message_clear()
            coder.history_message_clear()
            reviewer_func.history_message_clear()
            analyst_sec.history_message_clear()

            reports['code'].append(code)
            reports['func_plan'].append(plan)
            reports['func_review'].append(review)
            reports['sec_review'].append(new_risk)

    elif multi_agent == 'secAnalyst_coder_funcReviewer_secReviewer':

        analyst_sec = agents['analyst_sec']
        analyst_sec.update_requirement_code(requirement, code_unfinished)
        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)
        reviewer_func = agents['reviewer_func']
        reviewer_func.update_requirement_code(requirement, code_unfinished)

        analyst_sec.history_message_clear()
        coder.history_message_clear()
        reviewer_func.history_message_clear()


        for i in range(num_output):
            # Generate security risk
            analyst_sec_res = analyst_sec.task_potential_vulnerability(code_unfinished, requirement)
            
            risk = extract_before_fences(analyst_sec_res).strip()
            print('----------')
            print(risk)
            #plan = validate_plan(plan, analyst)

            # Generate code
            coder_chat = coder.task_coding(risk,topic='secure',do_sample=False,num_beams=1,temperature=1.0,top_p=1.0)
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content']).strip()

            reviewer_func_res = reviewer_func.task_review(code)
            review = extract_before_fences(reviewer_func_res).strip()
            review = extract_before_fences(review,fence='\n\n').strip()
            print('----------')
            print(review)
            analyst_sec_res = analyst_sec.task_potential_vulnerability(code, review)
            new_risk = extract_before_fences(analyst_sec_res).strip()

            analyst_sec.history_message_clear()
            coder.history_message_clear()
            reviewer_func.history_message_clear()

            reports['code'].append(code)
            reports['sec_plan'].append(risk)
            reports['func_review'].append(review)
            reports['sec_review'].append(new_risk)
    
    elif multi_agent == 'funcAnalyst_secAnalyst_coder_funcReviewer_secReviewer':

        analyst_func = agents['analyst_func']
        analyst_func.update_requirement_code(requirement, code_unfinished)
        analyst_sec = agents['analyst_sec']
        analyst_sec.update_requirement_code(requirement, code_unfinished)
        coder = agents['coder']
        coder.update_requirement_code(requirement, code_unfinished)
        reviewer_func = agents['reviewer_func']
        reviewer_func.update_requirement_code(requirement, code_unfinished)

        analyst_func.history_message_clear()
        analyst_sec.history_message_clear()
        coder.history_message_clear()
        reviewer_func.history_message_clear()


        for i in range(num_output):
            # Generate security risk
            analyst_func_res = analyst_func.task_requirement(context)
            analyst_sec_res = analyst_sec.task_potential_vulnerability(code_unfinished, requirement)
            
            plan = extract_before_fences(analyst_func_res).strip()
            risk = extract_before_fences(analyst_sec_res).strip()
            print('----------')
            print(plan)
            print(risk)
            #plan = validate_plan(plan, analyst)

            # Generate code
            coder_chat = coder.task_coding(plan+'\nBe aware of the following potential security risk and use mitigation when needed:\n'+risk,topic='plan',do_sample=False,num_beams=1,temperature=1.0,top_p=1.0)
            code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content']).strip()


            reviewer_func_res = reviewer_func.task_review(code)
            review = extract_before_fences(reviewer_func_res).strip()
            review = extract_before_fences(review,fence='\n\n').strip()
            print('----------')
            print(review)
            analyst_sec_res = analyst_sec.task_potential_vulnerability(code, review)
            new_risk = extract_before_fences(analyst_sec_res).strip()

            analyst_func.history_message_clear()
            analyst_sec.history_message_clear()
            coder.history_message_clear()
            reviewer_func.history_message_clear()

            reports['code'].append(code)
            reports['func_plan'].append(plan)
            reports['sec_plan'].append(risk)
            reports['func_review'].append(review)
            reports['sec_review'].append(new_risk)


    return reports
