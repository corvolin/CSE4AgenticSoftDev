import os
import time
import copy
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import torch


class base_agent(object):
    def __init__(self, model='codellama/CodeLlama-7b-Instruct-hf', majority=1, max_tokens=512, temperature=0.0, top_p=1.0):
        
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

    def get_history_message(self):
        return self.history_message
    
    def construct_system_message(self, role, team=''):
        if team == '':
            system_message = role
        else:
            system_message = role + '\n' + team
                    
        return system_message
