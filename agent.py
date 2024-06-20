from typing import Dict
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.memory.buffer import ConversationBufferMemory
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_functions
from langchain.agents.output_parsers.openai_functions import OpenAIFunctionsAgentOutputParser
from  langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function
import os
os.environ['http_proxy'] = "127.0.0.1:7890"
os.environ['https_proxy'] = "127.0.0.1:7890"
class Agent:
    def __init__(self, tools=None, question=None) -> None:
        self.temperature = 0.9
        self.tools = tools or []
        self.functions = []
        self.model = None
        self.prompt = None
        self.chain = None
        self.qa = None
        self.model_name = "gpt-4-1106-preview"
        self.question = question
        self.initialize_agent()

    def initialize_agent(self):
        # 加载和转换工具函数
        self.load_tools()
        # 初始化模型和其他组件
        self.initialize_components()

    def load_tools(self):
        # 假设 format_tool_to_openai_function 是一个已定义的函数
        # self.functions = [format_tool_to_openai_function(f) for f in self.tools]
        pass

    def initialize_components(self):
        # self.model = ChatOpenAI(temperature=self.temperature, model=self.model_name)
        self.model = AzureChatOpenAI(
            azure_deployment='gpt-4-32k-0613',
            api_version="2024-02-01",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.prompt = self.create_prompt()
        # self.chain = self.create_chain()
        self.agent = create_openai_functions_agent(self.model, self.tools, self.prompt)
        # agext=  AgentExecutor(agent=agent, tools=self.tools,verbose=True)
        self.qa = AgentExecutor(agent=self.agent,tools=self.tools,verbose= True,max_execution_time=120)
    def create_prompt(self):
        # 创建并返回 prompt
        # 这里省略具体实现，可以根据需要调整
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Your task is to fix the code based on the error message and return execution results.
You have three tools at your disposal,Please think about which tool to choose and pass in the required parameters:
The first is a code repair tool, which is used to detect code errors and fix them. you should pass code and error and question to the tool.;
The second is the code execution environment, used to test the repaired code.
The third is the result extractor, used to extract the answer after correct execution
If the code runs successfully, use the result extractor to collect the output. If it fails or produces unexpected results, go back to fix the code and repeat the process until you get satisfactory results.
Your goal is to process the provided code until it executes correctly and extract the output"""),
("user", "ALL data is as follows: \ncode:{code}\nerror:{error}\nquestion:{question}"),
            
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]).partial(question=self.question)
        return prompt
        

    def update_model(self, model_name, temperature=None):
        self.model_name = model_name
        if temperature is not None:
            self.temperature = temperature
        self.initialize_components()

    def add_tool(self, tool):
        self.tools.append(tool)
        self.initialize_agent()

    def __call__(self, input:Dict):
        # 结合原有的 invoke 方法
        return self.qa.invoke(input, return_only_outputs=True)
    
    def invoke(self, input_str):
        return self.qa.invoke(input_str, return_only_outputs=True)

    def get_messages(self):
        messages = {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith('_') and attr == "qa"}
        # 将messages中的每个条目格式化为更易于阅读的形式
        formatted_messages = {}
        for key, value in messages.items():
            # 这里是一个示例，根据属性类型进行不同的格式化
            if isinstance(value, list):
                formatted_value = "\n".join(str(item) for item in value)
            elif isinstance(value, dict):
                formatted_value = "\n".join(f"{k}: {v}" for k, v in value.items())
            else:
                formatted_value = str(value)
            formatted_messages[key] = formatted_value

        return formatted_messages
