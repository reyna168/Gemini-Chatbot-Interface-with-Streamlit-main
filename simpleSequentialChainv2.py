from langchain import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts import ChatPromptTemplate
from YiYan import YiYan

llm = YiYan()  # 文心目前只支持中英文

# prompt1: 将评论翻译成中文
prompt1 = ChatPromptTemplate.from_template(
    "将下面的评论翻译成中文:\n\n{Review}"
)
# chain 1: input= Review and output= Chinese_Review
chain_one = LLMChain(llm=llm, prompt=prompt1, output_key="Chinese_Review")

# prompt2: 概括评论
prompt2 = ChatPromptTemplate.from_template(
    "你能用 1 句话概括以下评论吗：\n\n{Chinese_Review}"
)
# chain 2: input= Chinese_Review and output= summary
chain_two = LLMChain(llm=llm, prompt=prompt2, output_key="summary")

# prompt3: 识别评论使用的语言
prompt3 = ChatPromptTemplate.from_template(
    "下面的评论使用的是什么语言？:\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=prompt3, output_key="language")

# prompt4: 生成特定语言的回复信息
prompt4 = ChatPromptTemplate.from_template(
    "使用指定语言编写对以下摘要的后续回复：\n\n摘要：{summary}\n\n语言：{language}"
)
# chain 4: input= summary,language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=prompt4, output_key="followup_message")

# prompt5: 将回复信息翻译成中文
prompt5 = ChatPromptTemplate.from_template(
    "将下面的评论翻译成中文:\n\n{followup_message}"
)
# chain 5: input= followup_message and output= Chinese_followup_message
chain_five = LLMChain(llm=llm, prompt=prompt5, output_key="Chinese_followup_message")

# chain: input= Review and output= language,Chinese_Review,summary,followup_message,Chinese_followup_message
chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four, chain_five],
    input_variables=["Review"],
    output_variables=["language", "Chinese_Review", "summary",
                      "followup_message", "Chinese_followup_message"],
    verbose=True
)

review = "I ordered a king size set. My only criticism would be that I wish seller would offer the king size set with 4 pillowcases. I separately ordered a two pack of pillowcases so I could have a total of four. When I saw the two packages, it looked like the color did not exactly match. Customer service was excellent about sending me two more pillowcases so I would have four that matched. Excellent! For the cost of these sheets, I am satisfied with the characteristics and coolness of the sheets."
print(review)
res = chain(review)
print(json.dumps(res, indent=4, ensure_ascii=False))
