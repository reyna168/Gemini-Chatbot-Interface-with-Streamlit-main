import os
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import ShellTool
from langchain import OpenAI
from YiYan import YiYan

os.environ["OPENAI_API_KEY"] = ''
llm = OpenAI(temperature=0)
# llm = YiYan()

shell_tool = ShellTool()
shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace("{", "{{").replace("}", "}}")

self_ask_with_search = initialize_agent(
    [shell_tool], llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
self_ask_with_search.run(
    # "下载网页https://huggingface.co的所有链接，只需要链接。并返回这些链接的排序列表只返回5条。必需使用双引号。"
    "Download https://huggingface.co webpage and grep for all urls. Return only a sorted list of them in size 5. Be sure to use double quotes."
)
