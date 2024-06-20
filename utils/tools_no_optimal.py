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
repl = PythonREPLTool()
@tool
def python_repl(code:Annotated[str,"python 代码"]):
    """excute the python code in the python repl tool"""
# 工具python_repl用来执行Python代码。如果你想要查看一个值的输出，应该使用`print(...)将其打印出来。这样用户就可以看到了。

    try:
        
        result = repl.run(code)
    except BaseException as e:
        return f"执行代码失败!错误:{repr(e)}"
    return f"成功执行python 代码:\n```python\n{code}\n```\nstdout: {result}"

# from langchain.tools import tool
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get solution."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.now()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'

@tool(return_direct=True)
def get_word_length(tool_input: str) -> int:
    """Returns the length of a word."""
    return len(tool_input)

# @tool
# def execute_data_and_code(data:str,code:str)->str:
#     """Execute the pyomo code script and return the result (correct result or error message)"""
    
#     full_code = f"{data}\n{code}\n"
#     # resource.setrlimit(resource.RLIMIT_CPU,(3,3))

#     # resource.setrlimit(resource.RLIMIT_AS,(1024*1024*500,1024*1024*500))

#     try:
#         res = subprocess.run(
#             ["python", "-c", full_code],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             check=True,
#             # capture_output=True
#             # text=True
#         )
#     except subprocess.SubprocessError as e:
#         err_message = str(e.stderr)
#         match = re.search("line(.*)", err_message)
#         if match:
#             err_message = match.group()
            
#             err_message = err_message.replace('\\n', '')
#             pattern = r'File ([^,]+), '
#             err_message = re.sub(pattern, '', err_message)
#         result = [{"key": "Error", "value": err_message}]
#         # result = "error:" + err_message
        
#     else:
#         result = res.stdout.decode("utf-8")
   
#     finally:
#         pass
#     return result
# 让代码在第一遍生成的时候就做一次检查

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
        output = res.stdout.decode("utf-8")
        try:
            result = extract_outputs(output)
        except ValueError as e:
            error_message = f"Failed to extract results: {e}"
            raise ValueError(error_message) from e
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
    # 检查是否至少传入了一个参数
    # 检查所有参数是否都已传入
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
    pattern1 = r'Total time[^\n]*\n(.*)'
    
    # 使用 re.DOTALL 标志进行搜索
    match = re.search(pattern1, output_str, re.DOTALL)
    
    # 若匹配成功，则返回匹配的内容且移除前后的空白字符
    if match:
        solver_output=match.group(1).strip()
        if not solver_output or solver_output=="":
            raise ToolException("There is no optimal solution for the current problem. Please try to modify the constraints or variable types to output an Infeasible solution.")
        else:
            return solver_output
    else:
        return  output_str
tool_extract_results = StructuredTool.from_function(
    func=extract_outputs,
    name="result_extractor",
    description="parser the details of the solver's execution and extract the part as output",
    handle_tool_error=True,
    # return_direct=True,
) 