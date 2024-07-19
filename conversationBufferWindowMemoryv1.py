from langchain import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from YiYan import YiYan

llm = YiYan()

memory = ConversationBufferWindowMemory(k=1)  # k=1,意味着只能记住最后1轮对话内容
conversation = ConversationChain(llm=llm,  memory=memory, verbose=True)

conversation.predict(input="你好，我叫周洁")
conversation.predict(input="可以推荐3首歌吗？")
conversation.predict(input="还记得我叫什么名字吗？顺便介绍下第一首歌的作者")
print(memory.buffer)

