import os
import time
import copy
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import torch
from agent.base_agent import base_agent


class python_developer(base_agent):
    def __init__(self, model='codellama/CodeLlama-7b-Instruct-hf', majority=1, max_tokens=512, temperature=0, top_p=1, 
                 team='',role='',requirement='', code_unfinished=''):
        super().__init__(model, majority, max_tokens, temperature, top_p)

        self.requirement = requirement
        self.code_unfinished = code_unfinished

        system_message = self.construct_system_message(role, team)
        self.history_message_append(system_message,'system')


    def task_coding(self, prompt, topic='', max_new_tokens=1024, do_sample=True, num_return_sequences=1, num_beams=1, temperature=0.05, top_p=0.95):
        
        if topic == '':
            input = prompt
        else:

            self.construct_with_report(prompt, topic)
            input = self.history_message
        
        try:
        
            outputs = self.model(
                input,  
                max_new_tokens=max_new_tokens,       
                num_return_sequences=num_return_sequences,
                do_sample=do_sample, 
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p
            )
            
            
            self.history_message_append(outputs[-1]['generated_text'][-1]['content'], "assistant")

            return outputs
        
        except torch.cuda.OutOfMemoryError as oom:
            # Free up GPU memory and advise user
            torch.cuda.empty_cache()
            print(f"CUDA OOM: {oom}. Try reducing max_new_tokens or switch to CPU.")

        except RuntimeError as rt_err:
            print(f"RuntimeError during generation: {rt_err}")

        except ValueError as val_err:
            print(f"ValueError: {val_err}. Check input types and template format.")

        except Exception as exc:
            # Catch-all for any other errors
            print(f"Unexpected error: {exc}")

        raise RuntimeError('Failed generator')
    
        
        
    def construct_with_report(self, report, topic):
        if topic == 'code':
            self.history_message_append(CODER_CODE.format(requirement=self.requirement,code=self.code_unfinished))
        elif topic == 'plan':
            self.history_message_append(CODER_PLAN.format(requirement=self.requirement,report=report.strip(),code=self.code_unfinished))
        elif topic == 'test':
            self.history_message_append(CODER_TEST.format(report=report.strip(),code=self.code_unfinished))
        elif topic == 'validate':
            self.history_message_append(CODER_VALIDATE.format(report=report.strip(),code=self.code_unfinished))
        elif topic == 'secure':
            self.history_message_append(CODER_SECURE.format(report=report.strip(),requirement=self.requirement,code=self.code_unfinished))
            
    def update_requirement_code(self, requirement, code_unfinished):
        self.requirement = requirement
        self.code_unfinished = code_unfinished
        
    
#CODER_CODE = 'The functional requirement of the Python function is:\n"""\n{requirement}\n"""\nComplete the following code:\n"""\n{code}\n"""\n```python'

CODER_CODE = '''
I need to implement a function in Python.
Please consider:
- Error handling
- Edge cases
- Performance optimization
- Best practices for Python
- Avoid vulnerability
Please do not unnecessarily remove any comments or code.
Let's think step by step and complete the code below with clear comments explaining the logic.
```python
{code}
"""
{requirement}
"""
```
'''
#CODER_PLAN = 'The plan from the requirement analyst is\n"""\n{report}\n"""\nComplete the code below\n"""\n{code}\n"""\n```python'
CODER_PLAN =  '''
Your job is to implement a function in Python.
Please consider:
- Error handling
- Edge cases
- Performance optimization
- Best practices for Python
- Avoid vulnerability
Please do not unnecessarily remove any comments or code.
Let's think step by step and complete the code below with clear comments explaining the logic.
```python
{code}
"""
{requirement}

{report}
"""
```
'''
CODER_TEST = 'The report from the tester is\n"""\n{report}\n"""\nUpdate the previous code\n```python'

CODER_VALIDATE = 'The code you provided previously is invalid due to\n"""\n{report}\n"""\nUpdate the previous code\n```python'

CODER_SECURE = '''
Your job is to implement a function in Python.
Please consider:
- Error handling
- Edge cases
- Performance optimization
- Best practices for Python
Please do not unnecessarily remove any comments or code.
Complete the code below with clear comments explaining the logic.
```python
{code}
"""
{requirement}
Be aware of the following potential security risk and use mitigation when needed:
{report}
"""
```
'''
    