import os
import time
import copy
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from agent.role_play import TESTER_TEST, TESTER_VALIDATE


class python_tester(object):
    def __init__(self, TEAM, python_tester, requirement, code_unfinished, model='codellama/CodeLlama-7b-Instruct-hf', majority=1, max_tokens=512,
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

        system_message = self.construct_system_message(python_tester, TEAM)
        self.history_message_append(system_message,'system')

    def task_testing(self, prompt, topic='', max_new_tokens=1024, do_sample=True, num_return_sequences=1, num_beams=1, temperature=0.05, top_p=0.95):
        
        self.construct_with_code_under_test(prompt, topic)
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
    
    def construct_with_code_under_test(self, report, topic):
        if topic=='test':
            self.history_message_append(TESTER_TEST.format(code=report,requirement=self.requirement))
        elif topic=='validate':
            self.history_message_append(TESTER_VALIDATE.format(report=report))
            
        
    def construct_system_message(self, role, team=''):
        if team == '':
            system_message = role
        else:
            system_message = role + '\n' +team
                    
        return system_message
    
    