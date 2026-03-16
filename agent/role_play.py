from string import Template

ANALYST = '''I want you to act as a requirement analyst on our development team. 
Given a unfinished Python code and its requirement in comment, your task is to decompose the requirement into concise plan to guide our developer to complete the unfinished code. 
The plan should be consistent with the context and only use the included API in the unfinished code .
Remember, you only need to provide a concise plan in markdown quote.
'''

SYS_ANALYST_PROBLEM = '''
You are a Requirements Engineer for a Python development team.
Your job is to:
1. Understand the functional requirement from the user.
2. Capture the context of the unfinished code.
3. Estimate the difficulty of given intent, check its clarity and lists its constraints.
Remember, you only need to the output the estimation of intent difficulty, clarity and constraints in ```json markdown quote.
'''

SYS_ANALYST_FUNCTIONALITY = '''
You are a Requirements Engineer for a Python development team.
your job is to:
1. Understand the functional requirement from the user.
2. Capture the context of the unfinished code.
3. Formulate the data and control flow to guide developers for implementing the desired functionality.
Remember, you only need to the output precise data and control flow in ```json markdown quote.
'''

SYS_REVIEWER_FUNCTIONALITY = '''
You are a Requirements Engineer for a Python development team.
your job is to:
1. Understand the functionality of a Python method.
2. Capture the context outside of the method.
3. Summarize the functionality of the method to guide developers for understanding the intention.
Remember, you only need to the output precise description of functionality in ```json markdown quote.
'''

SYS_ANALYST_SECURITY = '''
You are a Software Security expert for a Python development team.
your job is to:
1. Understand the functional requirement from the user.
2. Capture the context of the unfinished code.
3. Formulate a list of potential vulnerability and ways of mitigation to enhance code security without altering the desired functionality.
Remember, you only need to the output precise vulnerability and mitigation in ```json markdown quote.
'''

TESTER = '''I want you to act as a tester on our development team. You will receive the code written by the developer, and your job is as follows:
1. Write the test code that starts with "def check(candidate):" and candidate is a 'function' object.
2. Call candidate with different inputs (up to five) that starts with "print", and do not write assert statements.
Remember, you only need to provide the test code in Python and avoid using assert statements.
'''

PYTHON_DEVELOPER = '''I want you to act as a Python developer on our development team. You will receive unfinished code and plans from a requirement analyst or test reports from a tester. Your job is split into two parts:
1. If you receive a plan from a requirement analyst, complete code in Python that meets the requirement following the plan. Complete code snippets strictly within the provided context. Only use API from the provided unfinished code. Ensure that the code you write is efficient, readable, and follows best practices.
2. If you receive a test report from a tester, write the fixed or improved code based on the content of the report. Ensure that any changes made to the code do not introduce new bugs or negatively impact the performance of the code.
Remember, you need to provide the code in Python and explain the rational in comments. Use ```python to put the Python code in markdown quotes.
'''

PYTHON_TESTER = '''I want you to act as a tester on our development team. You will receive a single Python function written by the developer and functional requirement in plain text, and your job is as follows:
1. Write the Python Flask test suite that calls Python function with correct context.
2. Explain the reason of each test case in comment before their definition.
Remember, you need to provide the test code in Python and explain the rational in comments. Use ```python to put the Python test code in markdown quotes.
'''

PYTHON_REVIEWER = '''I want you to act as a security reviewer on our development team. You will receive a single Python function written by the developer and functional requirement in plain text, and your job is as follows:
1. Analyze the provided code and label potential vulnerability in the code with their type based on CodeQL.
2. For each vulnerability, provide instructions on how to mitigate them without changing functionality.
Remember, you need to provide the review in json. Use ```json to put the review in markdown quotes.
'''

TEAM_ACT = '''There is a development team that includes a requirement analyst, a Python developer, and a tester. The team needs to develop programs that satisfy the requirement of the users. The different roles have different divisions of labor and need to cooperate with each others.
'''

TEAM_FC = '''There is a development team that includes a functionality analyst, and a Python developer. The team needs to develop programs that satisfy the requirement of the users. The different roles have different divisions of labor and need to cooperate with each others.
'''
TEAM_FSC = '''There is a development team that includes a functionality analyst,a security analyst and a Python developer. The team needs to develop programs that satisfy the requirement of the users. The different roles have different divisions of labor and need to cooperate with each others.
'''

TEAM_SC = '''There is a development team that includes a security analyst, and a Python developer. The team needs to develop programs that satisfy the requirement of the users. The different roles have different divisions of labor and need to cooperate with each others.
'''

TEAM_CT = '''There is a development team that includes a Python developer and a tester. The team needs to develop programs that satisfy the requirement of the users. The different roles have different divisions of labor and need to cooperate with each others.
'''

TEAM_CF = '''There is a development team that includes a Python developer and a functionality reviewer. The team needs to develop programs that satisfy the requirement of the users. The different roles have different divisions of labor and need to cooperate with each others.
'''

TEAM_CR = '''There is a development team that includes a Python developer and a reviewer. The team needs to develop programs that satisfy the requirement of the users. The different roles have different divisions of labor and need to cooperate with each others.
'''

TEAM_FCF='''There is a development team that includes a functionality analyst, a Python developer and a functionality reviewer. The team needs to develop programs that satisfy the requirement of the users. The different roles have different divisions of labor and need to cooperate with each others.
'''

#ANALYST_REQUIREMENT = 'Here is the unfinished code """{code}""" and its function requirement is """{requirement}""". Use {size1} to {size2} words to formulate a concise plan in json for Python developer to complete the unfinished code that achieve the functional requirment\n```json\n'
ANALYST_REQUIREMENT = '''
I need to formulat a plan for implement a function in Python.
Functional requirement:
"""
{requirement}
"""
The context of the functions is:
```python
{code}
```
Please consider:
- What input to accept and output to return
- Divided overall functionality into smaller, manageable steps
- Depdency on import module and execution sequence
- Potential edge cases and exceptions
- Data flow and control flow
Be concise and use {size1} to {size2} words in markdown quote.
```plan
'''

ANALYST_VALIDATE = '''
Summary need to output as valid json.
Please use {size1} to {size2} words to summarize the plan in json markdown quotes.
```json
'''

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
Please do not unnecessarily remove any comments or code.
Complete the code below with clear comments explaining the logic.
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
TESTER_TEST = 'The functional requirement is:\n{requirement}\nNow use "unittest" liberary to generate concise and correct Python test suite in markdown quotes to verify the functional requirement.\n```python\n{code}\nclass TestSuite(unittest.TestCase):\n'
TESTER_VALIDATE = 'The test suite you provided previously is invalid due to:\n{report}\nPlease correct the Python test suite accordingly\n```python\n'

REVIEWER_CODE = '''
I need you to review a Python function and identify potential vulnerability if there is any.
```python
{code}
```
If the code is secure, your should output:
```json
{{
  "secure": true
}}
```
If the code contain vulnerability, you should output in the following format:
```json
{{
  "secure": false,
  "vulnerabilities":[{{
    "type": "type of vulnerability",
    "line": line number of vulnerability in the given code as Integer
    "fix": "recommendation on how to mitigate the vulnerability"
  }}]
}}
```
'''

REVIEWER_VALIDATE = '''
Your previous answer has the following issue;
{report}
Please format the previous answer in JSON only using the template below:
```json
{{
  "secure": false,
  "vulnerabilities":[{{
    "type": "type of vulnerability",
    "line": line number of vulnerability in the given code as Integer
    "fix": "recommendation on how to mitigate the vulnerability"
  }}]
}}
```
'''


ANALYST_REQUIREMENT_DCF = Template('''
### Task:
Your job is to describe the desired data and control flow of a unfinish function to fulfil its given functionality.
For data flow, you should describe data input, output, store and transformation.
For control flow, you should describe execution order, decision and loop.
Output your answer in JSON markdown quotes and generate only the quoted JSON.
### Example 1:
---
Unfinished code :
```python
from flask import Response, request, Flask
from werkzeug.datastructures import Headers
import re

app = Flask(__name__)

CONTENT_TYPES_GREENLIST = ['application/pdf', 'application/json', 'text/plain']

@app.route('/set_content_type')
def set_content_type():
    content_type = request.args.get('content_type', 'text/plain')
```
---
Desired functionality:
Get the user-specified content_type from the request,
content types should be allowed in the greenlist,
create a response with the content_type and send it as a response.
---
Output JSON:
```json
{
  "functionName": "set_content_type",
  "description": "Validate provided content_type against greenlist and return appropriate Flask Response",
  "inputs": {
    "content_type": {
      "type": "string",
      "source": "request.args.content_type",
      "default": "text/plain"
    }
  },
  "dataStores": {
    "greenlist": ["application/pdf", "application/json", "text/plain"]
  },
  "controlFlow": [
    {
      "condition": "inputs.content_type in dataStores.greenlist",
      "then": {
        "action": "buildResponse",
        "args": {
          "body": "\"OK\"",
          "status": 200,
          "headers": {
            "Content-Type": "{{inputs.content_type}}"
          }
        }
      },
      "else": {
        "action": "buildResponse",
        "args": {
          "body": "\"Invalid content type\"",
          "status": 400,
          "headers": {
            "Content-Type": "text/plain"
          }
        }
      }
    }
  ],
  "outputs": {
    "type": "Response",
    "properties": {
      "body": { "type": "string" },
      "status": { "type": "integer" },
      "headers": { "type": "object" }
    }
  }
}
```### Example 2:
---
Unfinished code :
```python
import tarfile
import os

def extract():
    with tarfile.open("archive.tar.gz") as tar:
```
---
Desired functionality:
extract the tar file to /tmp/unpack
go through each member of the tar file
all the extracted files should be under /tmp/unpack
---
Output JSON:
```json
{
  "functionName": "extract",
  "description": "Safely extract archive.tar.gz into /tmp/unpack, ensuring no path traversal",
  "inputs": {
    "archivePath": { "type": "string", "default": "archive.tar.gz" },
    "destination": { "type": "string", "default": "/tmp/unpack" }
  },
  "dataStores": {
    "members": { "type": "array", "items": "TarInfo" },
    "extracted": { "type": "array", "items": "string" }
  },
  "controlFlow": [
    {
      "action": "openTar",
      "args": { "path": "{{inputs.archivePath}}", "mode": "r:gz" },
      "resultVar": "tar"
    },
    {
      "action": "getMembers",
      "args": { "tar": "{{tar}}" },
      "resultVar": "dataStores.members"
    },
    {
      "type": "forEach",
      "collection": "dataStores.members",
      "itemVar": "member",
      "steps": [
        {
          "action": "computeTargetPath",
          "args": {
            "base": "{{inputs.destination}}",
            "memberName": "{{member.name}}"
          },
          "resultVar": "targetPath"
        },
        {
          "condition": "targetPath.startswith(inputs.destination + '/')",
          "then": [
            {
              "action": "extractMember",
              "args": {
                "tar": "{{tar}}",
                "member": "{{member}}",
                "path": "{{inputs.destination}}"
              }
            },
            {
              "action": "append",
              "args": {
                "list": "dataStores.extracted",
                "item": "{{member.name}}"
              }
            }
          ],
          "else": [
            {
              "action": "logWarning",
              "args": {
                "message": "Skipping unsafe member: {{member.name}}"
              }
            }
          ]
        }
      ]
    },
    {
      "action": "closeTar",
      "args": { "tar": "{{tar}}" }
    }
  ],
  "outputs": {
    "type": "object",
    "properties": {
      "extracted": { "type": "array", "items": "string" }
    },
    "required": ["extracted"]
  }
}
```
### Perform the given task on the following code and requirement:
---
Unfinished Code:
```python
$code
```
---
Desired functionality:
$requirement
---
Output JSON:
```json
''')


ANALYST_REQUIREMENT_ESTIMATE = Template('''
### Task:
Your job is to estimate the difficulty, check the clarity and list constraints of a unfinish function given its intent.
For task difficulty, you should estimate it as "easy", "medium" or "hard" from Python development perspective given the complexity of the task.
For intent clarity, you should answer "clear" or "ambiguous" depending on whether the intent clearly or vaguely describe the desired functionality.
For task constaints, you should answer "specific" or "general" depending on whether the intented functionality must be done through specific method or not.
Output your answer in JSON markdown quotes and generate only the quoted JSON.
### Example 1:
---
Unfinished code :
```python
from flask import Response, request, Flask
from werkzeug.datastructures import Headers
import re

app = Flask(__name__)

CONTENT_TYPES_GREENLIST = ['application/pdf', 'application/json', 'text/plain']

@app.route('/set_content_type')
def set_content_type():
    content_type = request.args.get('content_type', 'text/plain')
```
---
Desired functionality:
Get the user-specified content_type from the request,
content types should be allowed in the greenlist,
create a response with the content_type and send it as a response.
---
Output JSON:
```json
{
  "difficulty": "easy",
  "clarity": "clear",
  "constraint": "general"
}
```### Example 2:
---
Unfinished code :
```python
import tarfile
import os

def extract():
    with tarfile.open("archive.tar.gz") as tar:
```
---
Desired functionality:
extract the tar file to /tmp/unpack
go through each member of the tar file
all the extracted files should be under /tmp/unpack
---
Output JSON:
```json
{
  "difficulty": "medium",
  "clarity": "clear",
  "constraint": "specific"
}
```
### Perform the given task on the following code and requirement:
---
Unfinished Code:
```python
$code
```
---
Desired functionality:
$requirement
---
Output JSON:
```json
''')

ANALYST_SECURITY_CWE = Template('''
### Task:
Your job is to list the potential security vulnerabilities of a unfinished function and its desired functionality.
You should provide the CWE index number of the potential vulnerability, the rationale explaining why it is related to the unfinished function and way of mitigation.
Output your answer in JSON markdown quotes and generate only the quoted JSON.
### Example 1:
---
Unfinished code :
```python
from flask import Response, request, Flask
from werkzeug.datastructures import Headers
import re

app = Flask(__name__)

CONTENT_TYPES_GREENLIST = ['application/pdf', 'application/json', 'text/plain']

@app.route('/set_content_type')
def set_content_type():
    content_type = request.args.get('content_type', 'text/plain')
```
---
Desired functionality:
Get the user-specified content_type from the request,
content types should be allowed in the greenlist,
create a response with the content_type and send it as a response.
---
Output JSON:
```json
[
  {
    "cwe": 113,
    "rationale": "User-controlled 'content_type' is used to set the Content-Type header without validation against the greenlist, allowing HTTP header injection via CR/LF characters, leading to response splitting or malicious header insertion.",
    "mitigation": "Validate and strictly normalize the `content_type` against the whitelist. Reject input containing control characters (CR/LF). Prefer exact matching and encode or strip control characters."
  }
]
### Example 2:
---
Unfinished code :
```python
import tarfile
import os

def extract():
    with tarfile.open("archive.tar.gz") as tar:
```
---
Desired functionality:
extract the tar file to /tmp/unpack
go through each member of the tar file
all the extracted files should be under /tmp/unpack
---
Output JSON:
```json
[
  {
    "cwe": 22,
    "rationale": "Extracting a tar archive without sanitizing member filenames allows path traversal (\"tar-slip\"), enabling files to be written outside the intended directory (/tmp/unpack).",
    "mitigation": "Validate each member's resolved path. Before extraction, ensure that the normalized destination path starts with the intended base directory (e.g., using `os.path.realpath`), and skip entries with '..', absolute paths, or symlinks that point outside the target."
  }
]
### Perform the given task on the following unfinished code and desired functionality:
---
Unfinished Code:
```python
$code
```
---
Desired functionality:
$requirement
---
Output JSON:
```json
'''
)


ANALYST_SECURITY_SELF_SEC = Template('''
### Task:
Your job is to reivew code and predict if code is secure or not.
Output your answer either Yes or No in JSON markdown quotes and generate only the quoted JSON.
### Example 1:
---
code :
```python
import base64
import ujson
from flask import Flask, request, make_response, jsonify

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    raw_data = request.data
    raw_data = base64.b64decode(raw_data)
    try:
        data = ujson.loads(raw_data)
    except ValueError:
        return make_response(jsonify({"error": "Invalid input data"}), 400)
    return jsonify(data)
```
Output JSON:
```json
{
  "answer": "Yes"
}
### Example 2:
---
code :
```python
from flask import Flask, request
from lxml import etree
from lxml.etree import XMLParser

app = Flask(__name__)

@app.route('/get_root')
def get_root():
    xml_file_path = request.args.get('xml_file_path')
    # Validate the user-supplied file path to ensure it is a valid path to an XML file within the expected directory or subdirectory.
    if not xml_file_path or not xml_file_path.startswith('/expected/directory'):
        return 'Invalid file path', 400
    # Use a schema validator or an XML parser that supports schema validation to validate the XML schema before parsing the data.
    # Ensure that the schema is properly configured to prevent entity expansion attacks.
    schema_validator = etree.XMLSchema(file='/path/to/schema.xsd')
    parser = XMLParser(schema=schema_validator)
    try:
        tree = etree.parse(xml_file_path, parser=parser)
    except etree.XMLSyntaxError as e:
        return 'Invalid XML', 400
    root = tree.getroot()
    return etree.tostring(root, encoding='unicode')
```
Output JSON:
```json
{
  "answer": "No"
}
### Perform the given task on the following unfinished code and desired functionality:
---
Unfinished Code:
```python
$code
```
Output JSON:
```json
'''
)

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