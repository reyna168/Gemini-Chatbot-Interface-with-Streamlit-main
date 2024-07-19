from langchain.agents import tool, AgentType, initialize_agent, load_tools
from datetime import date
from YiYan import YiYan

llm = YiYan()

tools = load_tools(["llm-math", "wikipedia"], llm=llm)


@tool
def time(text: str) -> str:
    """
    Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function.
    """
    return str(date.today())


agent = initialize_agent(
    tools + [time], llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent("今天几月几日?")

