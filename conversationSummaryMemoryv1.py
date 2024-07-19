from langchain.memory import ConversationSummaryBufferMemory

schedule = """
上午 8 点与您的产品团队召开会议。\
您需要准备好幻灯片演示文稿。\
上午 9 点到中午 12 点有时间处理你的 LangChain 项目，\
这会进展得很快，因为 Langchain 是一个非常强大的工具。\
中午，在意大利餐厅与开车的顾客共进午餐\
距您一个多小时的路程，与您见面，了解人工智能的最新动态。\
请务必携带您的笔记本电脑来展示最新的LLM演示。\
"""

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

memory.save_context({"input": "你好"}, {"output": "什么事？"})
memory.save_context({"input": "没啥事情, 有个小问题请教"},
                    {"output": "好的，请说"})
memory.save_context({"input": "今天的日程安排是什么?"},
                    {"output": f"{schedule}"})
print(memory.load_memory_variables({}))
# {'history': 'System: The human greets the AI and asks about the daily schedule. The AI responds with a meeting in the morning and work on the LangChain project, followed by a lunch meeting with a customer who is driving over an hour to discuss AI updates. The human is asked to bring their laptop for a presentation.'}
