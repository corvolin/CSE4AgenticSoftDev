import os
import copy
import json
import argparse
import tqdm
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from agent.base_agent import base_agent

from agent.role_play import REVIEWER_CODE, REVIEWER_VALIDATE

class python_reviewer(base_agent):

    def __init__(self, model='codellama/CodeLlama-7b-Instruct-hf', majority=1, max_tokens=512, temperature=0.0, top_p=1.0,
                team='',role='',requirement=''):

        super().__init__(model, majority, max_tokens, temperature, top_p)

        self.requirement = requirement
        system_message = self.construct_system_message(role, team)
        self.history_message_append(system_message,'system')


    def task_review(self, prompt, topic='', max_new_tokens=1024, do_sample=True, num_return_sequences=1, num_beams=1, temperature=0.5, top_p=0.95):
        if topic == 'validate':
            self.history_message_append(REVIEWER_VALIDATE.format(report=prompt))
        elif topic == 'self_true':
            self.history_message_append(REVIEWER_VALIDATE.format(report=prompt))
        else:
            self.history_message_append(REVIEWER_CODE.format(code=prompt))
        input = self.history_message

        if topic == '':
            input = prompt
        else:
            self.construct_reviewer_prompt(prompt, topic)
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
    
    def construct_reviewer_prompt(self, prompt, topic):
        if topic == 'validate':
            self.history_message_append(REVIEWER_VALIDATE.format(report=prompt))
        elif topic == 'self_true':
            self.history_message_append(REVIEWER_VALIDATE.format(report=prompt))
        else:
            self.history_message_append(REVIEWER_CODE.format(code=prompt))