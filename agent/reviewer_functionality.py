import torch
from agent.base_agent import base_agent
from string import Template

class reviewer_functionality(base_agent):
    def __init__(self, model='codellama/CodeLlama-7b-Instruct-hf', majority=1, max_tokens=512,
                                temperature=0.0, top_p=1.0, team='',role=''):
        super().__init__(model, majority, max_tokens, temperature, top_p)
        

        system_message = self.construct_system_message(role, team)
        self.history_message_append(system_message,'system')


    def task_review(self, prompt, topic='', max_new_tokens=1024, do_sample=True, num_return_sequences=1, num_beams=1, temperature=0.5, top_p=0.95):
        
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
    
    def construct_with_report(self, report, topic):
        if topic == 'self_true':
            self.history_message_append(REVIEWER_FUNC_TRUE.substitute(requirement=report[0],code=report[1]))
        elif topic == 'func_eq':
            self.history_message_append(REVIEWER_FUNC_EQ.substitute(code1=report[0],code2=report[1]))
        else:
            self.history_message_append(REVIEWER_CODE_TO_FUNC.substitute(code=report))


REVIEWER_CODE_TO_FUNC = Template('''
### Task:
Your job is to describe the functionality behind the given Python method.
For functionality, you should describe it in 1 to 3 sentences using English lanugage.
Output your answer in JSON markdown quotes and generate only the quoted JSON.
### Example 1:
---
Complete code :
```python
from flask import Flask, request, send_file, send_from_directory
from werkzeug.utils import safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    if requested_file_name:
        file_path = safe_join(safe_dir, requested_file_name)
        return send_file(file_path)

    # send the requested files from the safe directory
    files = os.listdir(safe_dir)
    for file_name in files:
        yield send_file(safe_join(safe_dir, file_name))
```
---
Output JSON:
```json
{
  "functionality": "send the requested file from the safe directory"
}
```
### Example 2:
---
Complete code :
```python
import requests
from flask import Flask, request, Response

app = Flask(__name__)

@app.route("/visit")
def visit():
    subdomain = request.args.get('subdomain')
    domain = 'example.com'  # Allow specifying domain for testing

    # Validate the subdomain and domain to avoid security issues
    if not subdomain or not subdomain.isalnum():
        return Response("Invalid subdomain provided.", status=400)
    
    # Construct the URL safely
    url = f"https://{{subdomain}}.{{domain}}"

    try:
        # Make a GET request to the URL
        response = requests.get(url)
        # Return the content of the response
        return Response(response.content, status=response.status_code, mimetype=response.headers.get('Content-Type', 'text/plain'))
    except requests.exceptions.RequestException as e:
        # Handle exceptions that occur during the request
        return Response(str(e), status=500)

if __name__ == "__main__":
    app.run(debug=True)
```
---
Output JSON:
```json
{
  "functionality": "safely combine the subdomain with 'https://example.com' to have an url, make a request to the url, and return the response.\nonly allow the url request if the subdomain is one of the following: 'www', 'mail', 'blog'"
}
```
### Perform the given task on the following complete code:
---
Complete Code:
```python
$code
```
---
Output JSON:
```json
'''
)

REVIEWER_FUNC_EQ = Template('''
### Task:
Your are given two Python codes. Is there a functional equivalence between code1 and code2?
Output your answer either Yes or No in JSON markdown quotes and generate only the quoted JSON.
### Example 1:
---
Code1:
```python
def solution(lst):
  
  if not lst:
      raise ValueError("The input list cannot be empty.")
  
  total_sum = 0
  
  for i, num in enumerate(lst):
      if not isinstance(num, int):
          raise ValueError("All elements in the list must be integers.")
      
      if i % 2 == 0 and num % 2!= 0:
          total_sum += num
  
  return total_sum
```
---
Code2:
```python
def solution(lst):
  
  if not isinstance(lst, list):
      raise TypeError("Input must be a list.")
  
  if len(lst) == 0:
      raise ValueError("List cannot be empty.")
  
  sum_odd = 0
  
  for i in range(len(lst)):
      if i % 2 == 0 and lst[i] % 2!= 0:
          sum_odd += lst[i]
  
  return sum_odd
```
---
Output JSON:
```json
{
  "answer": "Yes"
}
```
### Example 2:
---
Code1:
```python
def largest_smallest_integers(lst):

  max_neg = None
  min_pos = None

  for num in lst:
      if not isinstance(num, int):
          raise ValueError("List contains non-integer value")

      if num < 0:
          if max_neg is None or num > max_neg:
              max_neg = num

      elif num > 0:
          if min_pos is None or num < min_pos:
              min_pos = num

  return (min_pos, max_neg)
```
---
Code2:
```python
def largest_smallest_integers(lst):

  largest_negative = None
  smallest_positive = None

  for num in lst:
      if not isinstance(num, int):  
          raise ValueError("List must contain only integers")
      
      if num < 0:  # Check for negative integers
          if largest_negative is None or num > largest_negative:
              largest_negative = num
      elif num > 0:  # Check for positive integers
          if smallest_positive is None or num < smallest_positive:
              smallest_positive = num

  return (largest_negative, smallest_positive)
```
---
Output JSON:
```json
{
  "answer": "No"
}
```
### Example 3:
---
Code1:
```python
def specialFilter(nums):

  if not isinstance(nums, list):
      raise ValueError("Input must be a list")
  
  if not all(isinstance(x, int) for x in nums):
      raise ValueError("All elements in the list must be integers")

  count = 0
  
  for num in nums:
      if num > 10:
          str_num = str(abs(num))  
          first_digit = int(str_num[0])
          last_digit = int(str_num[-1])
          
          if first_digit % 2!= 0 and last_digit % 2!= 0:
              count += 1
  
  return count
```
---
Code2:
```python
def specialFilter(nums):
  count = 0
  
  for num in nums:
      try:
          num_str = str(abs(num))  # Use abs to handle negative numbers
          
          if int(num_str) > 10:
              first_digit = int(num_str[0])
              last_digit = int(num_str[-1])
              
              if first_digit % 2!= 0 and last_digit % 2!= 0:
                  count += 1
  
      except ValueError:
          print(f"Warning: Non-integer value ignored.")
  
  return count
```
---
Output JSON:
```json
{
  "answer": "No"
}
```
### Perform the given task on the following complete code:
---
Code1:
```python
$code1
```
---
Code2:
```python
$code2
```
---
Output JSON:
```json
'''
)


REVIEWER_FUNC_TRUE = Template('''
### Task:
Your are given a functionality description and a Python Code. Does the code meet the functionality requirement?
Output your answer either Yes or No in JSON markdown quotes and generate only the quoted JSON.
### Example 1:
---
Desired Functionality:
Return list of all prefixes from shortest to longest of the input string
---
Code:
```python
from typing import List

def all_prefixes(string: str) -> List[str]:
  result = []

  for i in range(len(string)):
      result.append(string[:i+1])
  return result
```
---
Output JSON:
```json
{
  "answer": "Yes"
}
```
### Example 2:
---
Desired Functionality:
Return n-th Fibonacci number
---
Code:
```python
def fib(n: int):
  if n == 0:
    return 0
  if n == 1:
    return 2
  return fib(n - 1) + fib(n - 2)
```
---
Output JSON:
```json
{
  "answer": "No"
}
```
### Perform the given task on the following requirement and code:
---
Desired Functionality:
$requirement
---
Code:
```python
$code
```
---
Output JSON:
```json
'''
)