from langchain import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from YiYan import YiYan

llm = YiYan()

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
conversation = ConversationChain(llm=llm,  memory=memory, verbose=True)

conversation.predict(input="你好，我叫周洁")
conversation.predict(input="今天是几号？")
conversation.predict(input="还记得我叫什么名字吗？")
print(memory.buffer)