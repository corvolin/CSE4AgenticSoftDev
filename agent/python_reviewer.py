import os
import copy
import json
import argparse
import tqdm
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


from agent.role_play import REVIEWER_CODE, REVIEWER_VALIDATE

class python_reviewer(object):

    def __init__(self, TEAM, REQUIREMENT_ENGINEER, requirement, model='codellama/CodeLlama-7b-Instruct-hf', majority=1, max_tokens=512,
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
        self.history_message = []

        system_message = self.construct_system_message(REQUIREMENT_ENGINEER, TEAM)
        self.history_message_append(system_message,'system')

    def task_review(self, prompt, topic='', max_new_tokens=1024, do_sample=True, num_return_sequences=1, num_beams=1, temperature=0.5, top_p=0.95):
        if topic == 'validate':
            self.history_message_append(REVIEWER_VALIDATE.format(report=prompt))
        elif topic == 'self_true':
            self.history_message_append(REVIEWER_VALIDATE.format(report=prompt))
        else:
            self.history_message_append(REVIEWER_CODE.format(code=prompt))
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
        
    
    def construct_system_message(self, role, team=''):
        if team == '':
            system_message = role
        else:
            system_message = role + '\n' + team
                    
        return system_message