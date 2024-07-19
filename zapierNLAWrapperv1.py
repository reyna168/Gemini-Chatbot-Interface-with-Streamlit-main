from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from LLM.YiYan import YiYan
import os

os.environ["ZAPIER_NLA_API_KEY"] = 'sk-ak-xxxx'

llm = YiYan()
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 我们可以通过打印的方式看到我们都在 Zapier 里面配置了哪些可以用的工具
for tool in toolkit.get_tools():
    print(tool.name)
    print(tool.description)
    print("\n")

agent.run('请用中文总结最后一封发给我的邮件。并将总结发送给"xxx@gmail.com"')

