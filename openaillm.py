from langchain import OpenAI
import os
os.environ["OPENAI_API_KEY"] = ''  # 需要openai账号

# 创建OpenAI的LLM，默认为text-davinci-003, temperature控制结果随机程度，取值[0, 1]，越大越随机。
llm = OpenAI(temperature = 0.1)
print(llm('如何理解大模型的边界问题'))