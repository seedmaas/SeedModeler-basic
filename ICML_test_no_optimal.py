"""
Author: pengyi Zan
Date: 2024-05-26
Purpose: 不限制答案是否是Optimal的 可以容许线性松弛解输出 或者其他非optimal的解输出 1.解除optimality的限制 2.允许输出非最优解 其他逻辑与ICML_test一致
"""

from datetime import datetime 
import random
from langchain_core.pydantic_v1 import BaseModel, Field, validator, create_model # 注意包的信息create basemodel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnableLambda
import subprocess
import re
from langchain_core.prompts import load_prompt
import os
import json
from langchain_core.output_parsers import JsonOutputParser
from agent import Agent
from json import JSONDecodeError
from langchain_core.exceptions import OutputParserException
from utils.tools_no_optimal import tool_fix_bug,tool_execute_code,tool_extract_results
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts.chat import ChatPromptTemplate
from typing import Dict, Any
os.environ['http_proxy'] = "127.0.0.1:7890"
os.environ['https_proxy'] = "127.0.0.1:7890"
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


def execute_data_and_code(input_string):
    """ 执行提取的Python代码 抛出所有可能的异常"""
    if isinstance(input_string,dict):
        input_string=input_string["code"]
    else:
        input_string=input_string
    try:
        exc_code = extract_data_blocks(input_string)
        # print("Executing data and code:", exc_code)
    except ValueError as e:
        raise ValueError("Error: Missing data or code blocks.") from e

    try:
        res = subprocess.run(
            ["python", "-c", exc_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        err_message = e.stderr.decode('utf-8')  # 假设err_message是bytes类型
        # 直接抛出修改后的异常信息
        raise RuntimeError(f"Execution failed with error: {err_message}") from e
    else:
        output = res.stdout.decode("utf-8")
        try:
            result = extract_results_pass(output)
        except ValueError as e:
            error_message = f"Failed to extract results: {e}"
            raise ValueError(error_message) from e
        return result


def update_dict_with_results(dict1, result):
    """对于答案的parser 确保不会缺省任何一个key"""
    # 首先，设置所有dict1中的value为空字符串
    for key in dict1:
        dict1[key] = ""

    # 标记用于判断是否存在不在dict1中的键
    all_keys_valid = True

    # 遍历result字典，检查并更新dict1
    for result_key, value in result.items():
        # 如果key存在于dict1中，则更新值；否则，设置标记为False并立即结束循环
        if result_key in dict1:
            dict1[result_key] = value
        else:
            all_keys_valid = False
            break

    # 如果发现不在dict1中的键，将dict1所有值重置为空字符串
    if not all_keys_valid:
        for key in dict1:
            dict1[key] = ""

    return dict1



# 根据输出格式定义Pydantic BaseModel 以便统一输出的格式 方便校验
def create_output_model(data: Dict[str, Any]) -> BaseModel:
    fields = {
        key.replace(" ", "_"): (str, Field(default="", alias=key, description=key)) # 增加了一个description
        for key, _ in data.items()
    }
    # 动态创建并返回 Pydantic BaseModel，没有自定义Config 规范化输出格式
    return create_model('Result', **fields)

def model_to_mapping(model: BaseModel) -> Dict[str, Any]:
    properties = {}
    required = []

    # 构造properties信息
    for name, field in model.model_fields.items():
        properties[name] = {
            "type": "string",
            "description": field.description
        }
        if field.is_required:
            required.append(name)

    # 构造整体映射信息
    model_info = {
        'name': model.__name__,
        'description': "map numeric results to their semantically equivalent JSON keys appropriately",
        'parameters': {
            'type': 'object',
            'properties': properties,
            'required': required
        }
    }
    return model_info

def extract_results_pass(output_str):
    """处理求解结果 提取变量部分"""
    if isinstance(output_str,dict):
        output_str=output_str["output"]
    else:
        output_str=output_str
    pattern = r'Total time[^\n]*\n(.*)'
    
    # 使用 re.DOTALL 标志进行搜索
    match = re.search(pattern, output_str, re.DOTALL)
    
    # 若匹配成功，则返回匹配的内容且移除前后的空白字符
    if match:
        solver_output=match.group(1).strip()
        if not solver_output or solver_output=="":
            raise ValueError("There is no optimal solution for the current problem. Please try to modify the constraints or variable types to output an Infeasible solution.")
        else:
            return solver_output
    else:
        # 如果没有找到匹配，则返回原字符串
        return output_str

def execute_data_and_code_without_error(input_string):
    """ 执行提取的Python代码，并返回结果，不抛出异常 """
    try:
        exc_code = extract_data_blocks(input_string)
        # print("executing data and code",exc_code)
    except ValueError as e:
        print(e)

    # cleaned_code = clean_code(input_string)
    try:
        res = subprocess.run(
            ["python", "-c", exc_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        err_message = e.stderr.decode('utf-8')
        # 使用正则表达式解析错误信息
        result = err_message
        
    else:
        result = res.stdout.decode("utf-8")

    return result

def construct_chain(result_form: dict,question,model_version="gpt-4",save_run_to_json_definition=None,save_run_to_json_code=None):
    load_dotenv()
    model = AzureChatOpenAI(
            azure_deployment='gpt-4-32k-0613',
            api_version="2024-02-01",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )


    output_form = create_output_model(result_form)

    # prompt定义
    prompt_pulp_definition_path = "./PromptEngineering/pulp_definition/prompt_en.yaml"
    prompt_pulp_definition=load_prompt(prompt_pulp_definition_path)
    prompt_pulp_code_path="./PromptEngineering/pulp_no_optimal/prompt_en.yaml"
    prompt_pulp_code=load_prompt(prompt_pulp_code_path)

    json_string = json.dumps(result_form)
    escaped_json_string = json_string.replace("{", "{{").replace("}", "}}")
    prompt_result = ChatPromptTemplate.from_messages([
    ("system", f"Your task is to fill the valid solutions into a semantically similar JSON object structure. Your main tasks include the following steps:1. Extract data: Parse the solutions to obtain the valid values of the decision variables and objectives.2. Semantic mapping: Numeric results are appropriately mapped to their semantically equivalent JSON keys3. Error handling: You must fill in 'Error' in the value of the JSON object only if the solution indicates an error or None.4. Return Json object with values.The Json object is as follows:{escaped_json_string}"), 
    ("user", "{result}"),
    ])

    # 定义chain
    define_model_chain = prompt_pulp_definition | model | StrOutputParser().with_listeners(on_end=save_run_to_json_definition)
    generate_code_chain=prompt_pulp_code | model | StrOutputParser().with_listeners(on_end=save_run_to_json_code)
    execute_code_chain=RunnableLambda(func=execute_data_and_code)
    # generate_code_fallback=prompt_pulp_code | model | StrOutputParser()| RunnableLambda(func=execute_data_and_code_without_error)

    tools_list=[tool_execute_code,tool_fix_bug,tool_extract_results]
    agent=Agent(tools=tools_list,question=question)
    agent_fallback=agent.qa.with_fallbacks(
        exceptions_to_handle=(ValueError,RuntimeError,NameError,TypeError,subprocess.CalledProcessError), # Retry only on ValueError
        fallbacks=[RunnableLambda(func=execute_data_and_code_without_error)], # Retry with exponential backoff 如果agent出错后无法求解会将最开始输入给agent的部分直接求解得到答案
    )
    excute_code_fallback_chain = execute_code_chain.with_fallbacks(
            exceptions_to_handle=(ValueError,RuntimeError,NameError,TypeError,subprocess.CalledProcessError), # Retry only on ValueError
            fallbacks=[agent_fallback], # Retry with exponential backoff
            exception_key='error',
    )
    extract_results_chain=RunnableLambda(func=extract_results_pass)
    get_solution_chain = prompt_result | model | JsonOutputParser(pydantic_object=output_form).with_fallbacks(
        exceptions_to_handle=(OutputParserException,JSONDecodeError), # Retry only on ValueError
        fallbacks=[StrOutputParser()],
    )
    
    full_chain = (
            define_model_chain
            | (lambda input: {"input_variable": input})
            | generate_code_chain
            | (lambda input: {"code": input})
            | excute_code_fallback_chain
            | extract_results_chain
            | (lambda input: {"result": input})
            | get_solution_chain
            | RunnableLambda(func=wrap_values_with_string)
    )
    
    return full_chain


def wrap_values_with_string(data):
    # 检查传入的参数是否为字典类型
    if isinstance(data, dict):
        # 如果是字典类型，遍历字典并将所有的值转换为字符串
        for key in data:
            data[key] = str(data[key])
        return data
    else:
        # 如果不是字典类型，直接返回传入的参数
        return data


if __name__ == '__main__':

    file_path = 'docs/ICML/task3_test_public.json'
    batch_size = 25  # 设置每批次处理的大小
    batches_saved = 0  # 用于计算和记录保存的批次数量
    with open(file_path, "r") as f:
        train_data = json.load(f)
    processed_items = []
    log_file_path = "process_log_1.txt"
    with open(log_file_path, "w") as log_file:   
        for i,item in enumerate(train_data[402:]):
            id = item["id"]                                                                                                                                                                                                                                                   
            print(f"处理到 Processing item {id}")
            question = item["question"]
            data_example = item["results"]
            file_name = f"output_json/test_gpt4_no_optimal/{id}.json"
            
            # 将item保存到文件中
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=4)
            save_run_to_json_definition = make_save_run_to_json_definition(file_name)
            save_run_to_json_code=make_save_run_to_json_code(file_name)
            for key in data_example.keys():
                data_example[key] = ""

            full_chain = construct_chain(result_form=data_example,
                                        question=question,
                                        model_version="gpt-4",
                                        save_run_to_json_definition=save_run_to_json_definition,
                                        save_run_to_json_code=save_run_to_json_code,)
            keys_with_underscores = [key.replace(" ", "_").replace("'", "_") for key in data_example.keys()]

            inference = full_chain.invoke({"input": question,"key":keys_with_underscores})
            item["results"]= inference
            print(inference)
            # 首先，打开并读取现有的JSON文件
            with open(file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 向读取的数据中添加新的键值对
            data["results"] = inference

            # 然后，将更新后的数据保存回同一个文件    
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            if (i + 1) % batch_size == 0 or (i + 1) == len(train_data):
                batches_saved += 1
                save_path = file_path.replace('.json', f'_two_{batches_saved}.json')
                with open(save_path, "w") as f:
                    json.dump(train_data, f, ensure_ascii=False, indent=4)
                print(f'___________________________________________________________Saved batch {batches_saved} to {save_path}.')
        # 分批次保存
            
        # 最后保存到文件中
        final_save_path = file_path.replace('.json', '_final.json')
        with open(final_save_path, "w") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        print(f'Final data saved to {final_save_path}.') 


