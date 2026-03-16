import json 
import ast, builtins
import json
from typing import Set, List, Tuple, Union
import re
import tokenize
from io import BytesIO

def validate_plan(plan, analyst, validate_limit=2):
    validate_count = 0

    while validate_count<validate_limit:
        validate_msg = ''
        #print('---------validate plan-------------')
        #print(plan)
        try:
            json.loads(plan)
        except ValueError as e:
            validate_msg = 'Please provide the plan in valid json only.\n'
            
        # generate new plan if needed
        if validate_msg == '':
            break
        else:
            print('------------invalid plan------------')
            analyst_chat = analyst.task_requirement(validate_msg,topic='validate')
            new_plan = extract_between_fences(analyst_chat[-1]['generated_text'][-1]['content'])
            
            if plan == new_plan:
                print('------------invalid plan no longer improve------------')
                break
            else:
                plan = new_plan

        validate_count = validate_count+1
    return plan

def validate_review(review, reviewer, validate_limit=2):
    validate_count = 0
    secure_code = False

    while validate_count<validate_limit:
        validate_msg = ''
        #print('---------validate plan-------------')
        #print(plan)
        try:
            review_dict = json.loads(review)
        except ValueError as e:
            validate_msg = 'The review is not valid JSON'

        if validate_msg == '':
            if 'secure' not in review_dict:
                validate_msg = validate_msg + 'The review does not contain "secure" key\n'
            else:
                if review_dict['secure']:
                    # code is secure
                    secure_code = True
                    break
                else :
                    if 'vulnerabilities' not in review_dict:
                        validate_msg = validate_msg + 'The review does not contain "vulnerabilities" key\n'
                    else:
                        vulnerabilities = review_dict['vulnerabilities']
                        for i, vul in enumerate(vulnerabilities):
                            if 'type' not in vul:
                                validate_msg = validate_msg + 'The {0}th vulnerability in "vulnerabilities" does not contain "type"\n'.format(i)
                            if 'line' not in vul:
                                validate_msg = validate_msg + 'The {0}th vulnerability in "vulnerabilities" does not contain "line"\n'.format(i)
                            if 'fix' not in vul:
                                validate_msg = validate_msg + 'The {0}th vulnerability in "vulnerabilities" does not contain "fix"\n'.format(i)
            
        # generate new plan if needed
        if validate_msg == '':
            break
        else:
            print('------------invalid review------------')
            reviewer_chat = reviewer.task_review(validate_msg,topic='validate')
            new_review = extract_between_fences(reviewer_chat[-1]['generated_text'][-1]['content'])
            
            if review == new_review:
                print('------------invalid review no longer improve------------')
                break
            else:
                review = new_review

        validate_count = validate_count+1
    return review, secure_code

def validate_code(code, context, coder, validate_limit=2):
    
    validate_count = 0

    while validate_count<validate_limit:
        validate_count = validate_count+1
        validate_msg = ''
        
        # check python syntax
        try:
            ast.parse(code)
        except SyntaxError:
            validate_msg = 'Generated code contain syntax error.\n'

        # check context
        if validate_msg=='':
            validate_msg = check_API_dependency(code, context) + check_variable_definition(code)

        # generate new code if needed
        if validate_msg == '':
            break
        else:
            print('---------invalid code-------------\n', validate_msg)
            coder_chat = coder.task_coding(validate_msg,topic='validate')

        new_code = extract_between_fences(coder_chat[-1]['generated_text'][-1]['content'])
        # check if llm output freeze
        if code == new_code:
            print('------------invalid code no longer improve------------')
            break
        else:
            code = new_code
            
    return code

def validate_test(test, code, tester, validate_limit=2):
    validate_count = 0
    validate_msg = ''

    while validate_count<validate_limit:
        validate_count = validate_count+1
        
        # check python syntax
        try:
            ast.parse(test)
        except SyntaxError:
            validate_msg = 'Generated test contain syntax error.\n'
        if validate_msg=='':
            if not validate_test_case_class(test):
                validate_msg ='Generated test case should be a subclass of unittest.TestCase\n'
            uncovered_route = check_route_test_coverage(code, test)
            if len(uncovered_route)>0:
                validate_msg = validate_msg + 'Generated test suite should cover route: '+', '.join(uncovered_route)+'\n'

        if validate_msg == '':
            break
        else:
            print('---------invalid test-------------', validate_msg)
            tester_chat = tester.task_testing(validate_msg,topic='validate')

        new_test = extract_between_fences(tester_chat[-1]['generated_text'][-1]['content'])
        # check if llm output freeze
        if test == new_test:
            print('------------invalid test no longer improve------------')
            break
        else:
            test = new_test

    if not has_unittest_main_block(test):
        test = test + '\nif __name__ == "__main__":\n   unittest.main()\n'

    return test
def has_unittest_main_block(code_str):
    """
    Checks if the given Python code string contains:
    if __name__ == "__main__":
        unittest.main()
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        print(f"SyntaxError while parsing code: {e}")
        return False

    for node in tree.body:
        # Look for: if __name__ == "__main__":
        if isinstance(node, ast.If):
            # Check if the test is: __name__ == "__main__"
            test = node.test
            if (isinstance(test, ast.Compare) and
                isinstance(test.left, ast.Name) and test.left.id == '__name__' and
                len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq) and
                len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant) and
                test.comparators[0].value == '__main__'):
                # Check if unittest.main() is called in the body
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute):
                            if (isinstance(func.value, ast.Name) and func.value.id == 'unittest' and
                                func.attr == 'main'):
                                return True
    return False

def check_route_test_coverage(app_code: str, test_code: str) -> List[str]:
    """
    Returns a list of routes defined in the app that are not covered by any test case.
    """
    app_routes = extract_routes_from_app(app_code)
    tested_routes = extract_tested_routes(test_code)
    uncovered_routes = app_routes - tested_routes
    return list(uncovered_routes)

def extract_tested_routes(test_code: str) -> Set[str]:
    """
    Extracts all route paths that are tested in the test code.
    Assumes that tested routes are passed as strings to client.get or client.post.
    """
    tree = ast.parse(test_code)
    tested_routes = set()

    class TestRouteVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in {'get', 'post', 'put', 'delete', 'patch'}:
                    if node.args:
                        route_arg = node.args[0]
                        if isinstance(route_arg, ast.Str):
                            tested_routes.add(route_arg.s.split('?')[0])
            self.generic_visit(node)

    TestRouteVisitor().visit(tree)
    return tested_routes

def extract_routes_from_app(app_code: str) -> Set[str]:
    """
    Extracts all route paths from the Flask application code.
    """
    tree = ast.parse(app_code)
    routes = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                # Check for @app.route decorators
                if isinstance(decorator, ast.Call):
                    if hasattr(decorator.func, 'attr') and decorator.func.attr == 'route':
                        if decorator.args:
                            route_path = decorator.args[0]
                            if isinstance(route_path, ast.Str):
                                routes.add(route_path.s)
    return routes

def validate_test_case_class(code_str):
    """
    Parses the given Python code string and checks if the last class defined
    is a subclass of unittest.TestCase.

    Parameters:
    - code_str (str): The Python code as a string.

    Returns:
    - bool: True if the last class is a subclass of unittest.TestCase, False otherwise.
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(code_str)
    except SyntaxError as e:
        print(f"SyntaxError while parsing code: {e}")
        return False

    # Find all class definitions
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if not class_defs:
        print("No class definitions found in the code.")
        return False

    # Get the last class definition
    last_class = class_defs[-1]

    # Check if unittest is imported in the code
    imports_unittest = any(
        isinstance(node, (ast.Import, ast.ImportFrom)) and
        any(alias.name == 'unittest' for alias in node.names)
        for node in tree.body
    )

    if not imports_unittest:
        print("The code does not import the 'unittest' module.")
        return False

    # Check if the last class inherits from unittest.TestCase
    for base in last_class.bases:
        if isinstance(base, ast.Attribute):
            if (isinstance(base.value, ast.Name) and
                base.value.id == 'unittest' and
                base.attr == 'TestCase'):
                return True
        elif isinstance(base, ast.Name):
            if base.id == 'TestCase':
                return True

    return False

def extract_all_failures(error_str):
    """
    Extracts all failure messages from a unittest error output string.
    Returns a list of strings in the format:
    'test_function_name:AssertionError: error_message'
    """
    failures = []
    lines = error_str.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Identify the start of a failure block
        if line.startswith('FAIL:'):
            # Extract test function name
            match = re.match(r'FAIL: (\S+)', line)
            if match:
                test_name = match.group(1)
                # Initialize variables to store error details
                error_type = ''
                error_msg = ''
                # Traverse the subsequent lines to find AssertionError
                j = i + 1
                while j < len(lines):
                    current_line = lines[j].strip()
                    if current_line.startswith('FAIL:') or current_line.startswith('ERROR:'):
                        break  # Next failure block starts
                    if 'AssertionError' in current_line:
                        # Extract the error message
                        error_match = re.match(r'AssertionError: (.*)', current_line)
                        if error_match:
                            error_msg = error_match.group(1).strip()
                            error_type = 'AssertionError'
                            break
                    j += 1
                if error_type and error_msg:
                    # Remove spaces around '!=' for consistency
                    error_msg = re.sub(r'\s*!=\s*', '!=', error_msg)
                    failures.append(f"Test case: {test_name.replace('_',' ')} Error: {error_type} {error_msg}")
            i = j
        else:
            i += 1
    return failures

def extract_imports(code: str) -> Set[str]:
    """
    Parse the given Python code string and return a set of top-level module names it imports.
    Handles both 'import X' and 'from X import Y' forms.
    """
    tree = ast.parse(code)
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # alias.name may be 'os.path'; take top-level 'os'
                module = alias.name
                imports.add(module)
        elif isinstance(node, ast.ImportFrom):
            # node.module may be None (for relative imports); skip those
            if node.module:
                for alias in node.names:
                    # alias.name is the imported symbol
                    imports.add(alias.name)

    return imports


class UndefinedNameChecker(ast.NodeVisitor):
    def __init__(self):
        # names defined so far
        self.defined = set(dir(builtins))  # include built-ins
        self.undefined = set()                # list of (line, name) errors

    def visit_Import(self, node):
        for alias in node.names:
            # import x as y defines y or x if no alias
            name = alias.asname or alias.name.split('.')[0]
            self.defined.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            # from module import x as y defines y or x
            name = alias.asname or alias.name
            self.defined.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.defined.add(node.name)
        for arg in node.args.args:
            self.defined.add(arg.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # multiple targets: a = b = …
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined.add(target.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.defined:
                # undefined usage
                self.undefined.add((node.id, node.lineno))
        # no need to call generic_visit for Name
       
    def visit_Attribute(self, node):
        # descend to the base of chained attributes
        base = node
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name) and isinstance(base.ctx, ast.Load):
            if base.id not in self.defined:
                self.undefined.add((base.id, node.lineno))
        self.generic_visit(node)

def check_variable_definition(code: str):
    tree = ast.parse(code)
    checker = UndefinedNameChecker()
    checker.visit(tree)
    if checker.undefined:
        name_at_line = []
        for name in checker.undefined:
            name_at_line.append(name[0]+" at line "+str(name[1]))
        return "Generated code contain undefined name at corresponding line number: "+", ".join(name_at_line)+ "\n"
    else:
        return ''

def check_API_dependency(code: str, context: str):
    code_imports = extract_imports(code)
    context_imports = extract_imports(context+'\ndef main():\n    return True')

    unique_to_code = code_imports - context_imports
    unique_to_context = context_imports - code_imports
    # Format report
    lines = []
    if unique_to_code:
        lines.append("Generated code used external API and must delete them:" + ", ".join(unique_to_code) + "\n")
        
    if unique_to_context:
        lines.append("Generated code neglected provided API and must add them:" + ", ".join(unique_to_context) + "\n")
    
    return "\n".join(lines)

def find_method_name(code, lang="python"):
    try:
        parsed = ast.parse(code)
        function_defs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        if function_defs:
            if len(function_defs) == 1:
                method_name = function_defs[0].name
            else:
                method_name = function_defs[-1].name if function_defs[-1].name != "main" else function_defs[-2].name
        else:
            method_name = ''
    except:
        method_name = ''

    return method_name

def extract_all_comments(
    source: Union[str, bytes],
    from_file: bool = False
) -> Tuple[List[str], List[str]]:
    # 1. Load source bytes
    if from_file:
        with open(source, "rb") as f:
            code_bytes = f.read()
    else:
        code_bytes = source.encode("utf-8") if isinstance(source, str) else source

    line_comments: List[str] = []
    block_comments: List[str] = []

    # 2. Tokenize and collect
    tokens = tokenize.tokenize(BytesIO(code_bytes).readline)
    for toknum, tokval, *_ in tokens:
        # 2a. Line comments
        if toknum == tokenize.COMMENT:
            cleaned = tokval.lstrip('#').lstrip(' ').rstrip()
            if cleaned:
                line_comments.append(cleaned)
        # 2b. String literals that are triple-quoted
        elif toknum == tokenize.STRING:
            # tokval includes quotes; check for triple-quote delimiters
            if (tokval.startswith('"""') and tokval.endswith('"""')) or \
               (tokval.startswith("'''") and tokval.endswith("'''")):
                # Strip the three quotes at each end and unindent
                inner = tokval[3:-3]
                # Remove possible leading/trailing newlines/spaces
                cleaned = inner.strip()
                if cleaned:
                    block_comments.append(cleaned)
    output = ''
    for line in line_comments:
        output = output+line+'\n'
    for block in block_comments:
        output = output+block +'\n'
    return output

def remove_comments(code: str) -> str:
    """
    Remove all line comments and standalone block comments from Python code,
    without inserting the 'utf-8' encoding token at the top.
    """
    output_tokens = []
    reader = BytesIO(code.encode('utf-8')).readline

    prev_end = (1, 0)
    for tok in tokenize.tokenize(reader):
        tok_type, tok_string, start, end, _ = tok

        # Skip the encoding token, comments, and standalone triple-quoted strings
        if tok_type in (tokenize.ENCODING, tokenize.COMMENT):
            continue
        if tok_type == tokenize.STRING and (
            (tok_string.startswith('"""') and tok_string.endswith('"""')) or
            (tok_string.startswith("'''") and tok_string.endswith("'''"))
        ):
            continue

        # Preserve original spacing/newlines
        
        if prev_end[1] != start[1]:
            output_tokens.append(" " * (start[1] - prev_end[1]))

        output_tokens.append(tok_string)
        prev_end = end

    return "".join(output_tokens)

def extract_between_fences(text: str) -> str:
    """
    Return the substring between the first and second occurrence of '```'.
    
    If fewer than two '```' fences are found, returns an empty string.
    """
    fence = "```"
    # 1. Find first fence
    first = text.find(fence)
    if first == -1:
        return text # No first fence

    # 2. Find second fence, starting just after the first one
    second = text.find(fence, first + len(fence))
    if second == -1:
        second = len(text)

    # 3. Extract and return the content in between
    start = text.find('\n', first + len(fence)) + 1
    return text[start:second]

def extract_before_fences(text: str, fence = "```") -> str:
    # 1. Find first fence
    if fence+'json' in text:
        first = text.find(fence+'json')
        text = text[first+len(fence+'json'):]
    if fence+'python' in text:
        first = text.find(fence+'python')
        text = text[first+len(fence+'python'):]
    last = text.find(fence)
    if last == -1:
        return text 
    else:
        return text[0:last]

import tempfile
import subprocess
import os
from typing import Tuple

def run_generated_test(code: str, python_executable: str = "python") -> Tuple[int, str, str]:
    # 1. Create a temp file
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        # 2. Write the code to it
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(code)

        # 3. Run the temporary script
        completed = subprocess.run(
            [python_executable, path],
            capture_output=True,
            text=True,
            timeout=20
        )

        # 4. Return exit code and outputs
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as e:
        return 0,'','ERROR: Code correctness unknown due to test timeout'

    finally:
        # 5. Always delete the temp file
        try:
            os.remove(path)
        except OSError:
            pass