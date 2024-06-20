import json
import re
from typing import Annotated
import wikipedia
from langchain_core.pydantic_v1 import BaseModel, Field
import datetime
from langchain_core.tools import tool,BaseTool
# from langchain.agents import tool
import requests
import resource
import subprocess
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools  import ToolException
from langchain_core.tools import StructuredTool

def make_save_run_to_json_definition(filename):
    def save_run_to_json_definition(run_obj):
        # 这里我们假设run_obj是一个字典，有一个'outputs'的键，而且这个键关联的值也是一个字典，有一个'output'的键
        new_definition = run_obj.outputs['output']
        # print(new_definition)
        try:
            # 尝试打开并读取现有的JSON文件
            with open(filename, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
                # 检查现有的data是不是一个字典
                if not isinstance(existing_data, dict):
                    raise ValueError("Existing data is not a valid dictionary.")
        except (FileNotFoundError, ValueError):
            # 如果文件不存在或数据格式不对，则创建一个新的结构
            existing_data = {
                "id": 0,  # 根据需要可能要调整这个默认值
            }

        # 在现有数据字典中添加或更新"definition"字段
        existing_data["definition"] = new_definition

        # 将更新后的数据写回文件
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

    return save_run_to_json_definition


def make_save_run_to_json_code(filename):
    def save_run_to_json_code(run_obj):
        # 这里我们假设run_obj是一个字典，有一个'outputs'的键，而且这个键关联的值也是一个字典，有一个'output'的键
        new_code = run_obj.outputs['output']
        # print(new_code)
        try:
            # 尝试打开并读取现有的JSON文件
            with open(filename, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
                # 检查现有的data是不是一个字典
                if not isinstance(existing_data, dict):
                    raise ValueError("Existing data is not a valid dictionary.")
        except (FileNotFoundError, ValueError):
            # 如果文件不存在或数据格式不对，则创建一个新的结构
            existing_data = {
            }

        # 在现有数据字典中添加或更新"definition"字段
        code_lines = new_code.strip().split('\n')
        existing_data["code"] = code_lines

        # 将更新后的数据写回文件
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)

    return save_run_to_json_code



