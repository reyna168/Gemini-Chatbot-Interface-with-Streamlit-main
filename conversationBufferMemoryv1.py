from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from YiYan import YiYan

llm = YiYan()

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm,  memory=memory, verbose=True)

conversation.predict(input="你好，我叫周洁")
conversation.predict(input="可以推荐3首歌吗？")
conversation.predict(input="还记得我叫什么名字吗？")
print(memory.buffer)
print(memory.load_memory_variables({}))
# {'history': 'Human: 你好，我叫周洁\nAI: 你好，周洁，很高兴认识你。请问有什么我可以帮助你的吗？\nHuman: 可以推荐3首歌吗？\nAI: 当然可以，以下是我为您推荐的3首歌曲：\n1. 《平凡之路》- 朴树。这首歌的旋律简单，歌词深情，表达了平凡人生的意义和追求。\n2. 《光年之外》- 邓紫棋。这首歌的旋律动感，歌词充满力量，非常适合在心情低落时聆听。\n3. 《告白气球》- 周杰伦。这首歌的旋律浪漫，歌词甜蜜，非常适合送给心爱的人作为表白礼物。\n希望这些歌曲能够给您带来愉悦的心情和美好的体验。\nHuman: 还记得我叫什么名字吗？\nAI: 当然，我记得您叫周洁。如果您有任何需要帮助的地方，请随时告诉我。'}

memory = ConversationBufferMemory()
memory.save_context({"input": "你好"}, 
                    {"output": "有啥事吗?"})
print(memory.buffer)



