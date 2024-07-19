from langchain import PromptTemplate

template = "为{product}行业的公司起一个好名字，并给出原因"
prompt = PromptTemplate(input_variables=["product"], template=template)

input_text = '糖果'
print(prompt.format(product=input_text))  # 为糖果行业的公司起一个好名字，并给出原因