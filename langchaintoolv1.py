import os
from langchain.agents import load_tools, initialize_agent, AgentType
from YiYan import YiYan
os.environ['SERPAPI_API_KEY'] = ''  # 用谷歌账号申请一个免费api

llm = YiYan()

# 申请工具集
tools = load_tools(["serpapi", "llm-math", "wikipedia"], llm=llm)
# 初始化agent
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

question = "谁是现在的美国总统? 他的身高减去20是多少?"
res = agent(question)
print(res)