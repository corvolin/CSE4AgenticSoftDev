import os
import time
import copy
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import torch
from agent.role_play import CODER_PLAN, CODER_TEST, CODER_CODE, CODER_VALIDATE, CODER_SECURE


class python_developer(object):
    def __init__(self, TEAM, PYTHON_DEVELOPER, requirement, code_unfinished, model='codellama/CodeLlama-7b-Instruct-hf', majority=1, max_tokens=512,
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
        self.history_message = []
        self.requirement = requirement
        self.code_unfinished = code_unfinished

        if PYTHON_DEVELOPER != '':
            system_message = self.construct_system_message(PYTHON_DEVELOPER, TEAM)
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
        
    def construct_system_message(self, role, team=''):
        if team == '':
            system_message = role
        else:
            system_message = role + '\n' +team
                    
        return system_message
    
    