from langchain.llms import HuggingFaceHub
import os

os.environ['Huggacehub_API_TOKEN'] = ''
llm = HuggingFaceHub(repo_id ="THUDM/chatglm-6b")

print(llm('如何理解大模型的边界问题'))