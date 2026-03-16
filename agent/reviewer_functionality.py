import os
import copy
import json
import argparse
import tqdm
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from agent.role_play import REVIEWER_CODE_TO_FUNC, REVIEWER_FUNC_EQ, REVIEWER_FUNC_TRUE

class reviewer_functionality(object):
    def __init__(self, TEAM, REVIEWER_FUNCTIONALITY, requirement, code_unfinished, model='codellama/CodeLlama-7b-Instruct-hf', majority=1, max_tokens=512,
                                temperature=0.0, top_p=1.0):
        self.model = pipeline(
            "text-generation",
            model=model,
            tokenizer=model,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            torch_dtype=torch.float16,
            return_full_text=True
        )
        self.majority = majority
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.requirement = requirement
        self.code_unfinished = code_unfinished
        self.history_message = []

        system_message = self.construct_system_message(REVIEWER_FUNCTIONALITY, TEAM)
        self.history_message_append(system_message,'system')


    def task_review(self, prompt, topic='', max_new_tokens=1024, do_sample=True, num_return_sequences=1, num_beams=1, temperature=0.5, top_p=0.95):
        if topic == 'self_true':
            self.history_message_append(REVIEWER_FUNC_TRUE.substitute(requirement=prompt[0],code=prompt[1]))
        elif topic == 'func_eq':
            self.history_message_append(REVIEWER_FUNC_EQ.substitute(code1=prompt[0],code2=prompt[1]))
        else:
            self.history_message_append(REVIEWER_CODE_TO_FUNC.substitute(code=prompt))

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

            if topic == 'self_true':
                res = outputs[-1]['generated_text'][-1]['content'].removeprefix(REVIEWER_FUNC_TRUE.substitute(requirement=prompt[0],code=prompt[1]))
            elif topic == 'func_eq':
                res = outputs[-1]['generated_text'][-1]['content'].removeprefix(REVIEWER_FUNC_EQ.substitute(code1=prompt[0],code2=prompt[1]))
            else:
                res = outputs[-1]['generated_text'][-1]['content'].removeprefix(REVIEWER_CODE_TO_FUNC.substitute(code=prompt))
            
            return res
        
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
    
    def history_message_append(self, system_message, role="user"):
        self.history_message.append({
            "role": role,
            "content": system_message
        })
        
    def history_message_clear(self):
        sys_msg = self.history_message[0]
        self.history_message = []
        self.history_message.append(sys_msg)
        torch.cuda.empty_cache()
        
    def update_requirement_code(self, requirement, code_unfinished):
        self.requirement = requirement
        self.code_unfinished = code_unfinished

    def construct_system_message(self, role, team=''):
        if team == '':
            system_message = role
        else:
            system_message = role + '\n' + team
                    
        return system_message

