from langchain import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts import ChatPromptTemplate

from YiYan import YiYan

llm = YiYan()

# prompt template 1
prompt1 = ChatPromptTemplate.from_template(
    "描述生产{product}的公司的一个最佳名称是什么？只返回一个答案，答案限定为3到5字。"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=prompt1)

# prompt template 2
prompt2 = ChatPromptTemplate.from_template(
    "为以下公司编写 20 个字的描述：{company_name}”"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=prompt2)

# 将chain1和chain2组合在一起生成一个新的chain.
chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
# 执行新的chain
input_text = '糖果'
chain.run(input_text)

