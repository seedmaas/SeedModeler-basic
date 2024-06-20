import re
from typing import Annotated
import wikipedia
from langchain_core.pydantic_v1 import BaseModel, Field
import datetime
from langchain_core.tools import tool,BaseTool
import os
import requests
import resource
import subprocess
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools  import ToolException
from langchain_core.tools import StructuredTool
os.environ['http_proxy'] = "127.0.0.1:7890"
os.environ['https_proxy'] = "127.0.0.1:7890"

class CodeSchema(BaseModel):
    code: str  # 定义 query 字段为字符串类型

def extract_data_blocks(input_string):
        """ 从生成代码的输出中提取可以执行的Python代码部分 """
        python_pattern = r'```python(.*?)```'
        
        # 使用re.search查找匹配的代码块
        
        python_match = re.search(python_pattern, input_string, re.DOTALL)
        
        # 提取匹配的内容，如果没有找到，则为None
    
        python_code = python_match.group(1) if python_match else None
        
        # 如果json_data或python_code其中一个为None，则抛出KeyError异常
        if python_code is None:
            raise ValueError("Error: Missing data or code blocks.")
        
        execute_python_code = f"{python_code}"
        
        # 返回组合后的字符串
        return execute_python_code


def execute_pulp_code(code:str):
    """ 执行可运行的完整Python代码 抛出所有可能的异常"""
    if isinstance(code, dict):  # 检查是否为字典类型
        # 尝试获取字典中名为'code'的键对应的值
        if 'code' in code:
            code=code['code']

    if not code:
        raise ToolException("Error: Missing code input.")
        
    python_pattern = r'```python(.*?)```'
        
        # 使用re.search查找匹配的代码块
        
    python_match = re.search(python_pattern, code, re.DOTALL)
        
        # 提取匹配的内容，如果没有找到，则为None
    
    python_code = python_match.group(1) if python_match else code
    try:
        res = subprocess.run(
            ["python", "-c", python_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    # except subprocess.CalledProcessError as e:
    except Exception as e:
        err_message = e.stderr.decode('utf-8')  # 假设err_message是bytes类型
        # 直接抛出修改后的异常信息
        raise ToolException(f"Execution failed with error: {err_message}") from e
    else:
        result = res.stdout.decode("utf-8")
        if "No optimal solution" in result:
            raise ToolException(f"This code cannot generate a valid solution.Please check the logic of this code to see if it accurately interprets the intent behind the optimization question.")
       
        return result

tool_execute_code = StructuredTool.from_function(
    func=execute_pulp_code,
    name="execute_pulp_code",
    description="Receives executable Python code, runs the code, returns a successful result or raises an error",
    handle_tool_error=True,
) 

class FixBugInput(BaseModel):
    "question: str, error: str, code: str"
    question: str = Field("", description="optimization question")
    error: str = Field("",description="error")
    code: str = Field("",description="code")

def fix_bug(question: str = None, error: str = None, code: str = None) -> str:
    """Repair the code by taking the problem, code, and error as input and return the complete executable code."""

    print("fix_bug________________________________________________________________________________")
    print(question, error, code)
    if None in [question, error, code]:
        missing_params = [param_name for param_name, param_value in [('question', question), ('error', error), ('code', code)] if param_value is None]
        missing_params_str = ", ".join(missing_params)
        raise ToolException(f"Miss input: {missing_params_str}. Check the fix_bug tool needs question, error, code as input.")

    # 检查传入的参数是否为字符串类型和非空
    for arg_name, arg_value in [('question', question), ('error', error), ('code', code)]:
        # 先检查类型
        if not isinstance(arg_value, str):
            raise ToolException(f"The input param {arg_name} is not a string.")
        # 再检查是否为空
        if arg_value == "":
            raise ToolException(f"The input param {arg_name} is empty.")
        
    model = AzureChatOpenAI(
        azure_deployment=os.getenv('AZURE_OPENAI_GPT4_API_DEPLOYMENT_NAME'),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    prompt = ChatPromptTemplate.from_messages([
            ("system", """Your responsibility is to fix the code based on the error message. The language used in this code is PuLP, which is used to solve optimization problems. The following is the error message of the code and the optimization problem solved by this code.
Code: ```{code}```
Error message: {error}
Optimization problem: {question}
Just return the complete code wrapped in ```python```, don't return partial code, don't add any comments or explanations.
The necessary steps and corresponding examples are below. Please perform the following in order, replacing the corresponding variable names in code with the actual values in the question:
    Please ensure that your Python code can execute smoothly based on understanding the overall algorithm flow. Only output python code.
## Additional tips
Refer to the following common errors and corresponding solutions to fix your code:
0. TypeError: '<' not supported between instances of 'int' and 'LpAffineExpression': You need to rewrite the relevant expressions instead of directly using Python's calculation function
1. TypeError: '>' not supported between instances of 'LpVariable' and 'LpVariable': You may have used '>' or '<' incorrectly when adding constraints. This is not supported by PULP. Please replace '>' or ' <' is changed to '>=' or '<='.
2. Incorrectly defining the objective function as LpVariable without correctly defining its operation expression.
3. Name 'XXX' is not defined: An undefined variable appears in the code. For example, if "x" in "prob = x" is not previously defined, fix the code based on optimization issues.
4. Check whether the definition of the 'LpProblem' object is correct,  whether the LpProblem sense of LpMinimize or LpMaxmize meets the solution objective of the optimization problem
5. TypeError: unsupported operand type(s) for /: 'int' and 'LpVariable':Represent division by multiplying the reciprocal of a number.
""")])
    
    response = (prompt| model).invoke({"error":error,"code":code,"question":question})
    return response.content


def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
    )
    
tool_fix_bug = StructuredTool.from_function(
    func=fix_bug,
    args_schema=FixBugInput,
    name="fix_bug",
    description="Receives code, question, and error as input and return the complete executable code",
    handle_tool_error=_handle_error,
) 



def extract_outputs(output_str):
    """parser the details of the solver's execution and extract the part as output"""
    if isinstance(output_str,dict):
        first_key = next(iter(output_str))
        output_str = output_str[first_key]
    else:
        output_str=output_str
    if "No optimal solution" in output_str:
        raise ToolException("This code cannot generate a valid solution.Please check the logic of this code to see if it accurately interprets the intent behind the optimization question.")

    pattern1 = r'Total time[^\n]*\n(.*)'
    
    match = re.search(pattern1, output_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return  output_str
    
    

tool_extract_results = StructuredTool.from_function(
    func=extract_outputs,
    name="result_extractor",
    description="parser the details of the solver's execution and extract the part as output",
    handle_tool_error=True,
    # return_direct=True,
) 