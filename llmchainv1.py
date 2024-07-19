from langchain import PromptTemplate, LLMChain
from YiYan import YiYan

template = "描述生产{product}的公司的一个最佳名称是什么？只返回一个答案，答案限定为3到5字。"
prompt = PromptTemplate(input_variables=["product"], template=template)

llm = YiYan()
llm_chain = LLMChain(prompt=prompt, llm=llm)

input_text = '糖果'
print(prompt.format(product=input_text))
print(llm_chain.run(input_text))
