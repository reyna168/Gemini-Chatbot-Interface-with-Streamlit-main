from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain
from YiYan import YiYan

examples = [
    {"word": "happy", "antonym": "悲伤"},
    {"word": "fast", "antonym": "慢"},
]

example_template = """
单词: {word}
反义词: {antonym}
"""

prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    prefix="给出以下每个单词的反义词并翻译为中文",
    suffix="单词: {input}\n反义词:",
    input_variables=["input"],
    example_separator="\n",
)

input_text = 'big'
print(few_shot_prompt.format(input=input_text))

# 以下是结合LLMChain的使用样例
llm = YiYan()
llm_chain = LLMChain(prompt=few_shot_prompt, llm=llm, verbose=True)  # verbose=True显示中间过程
print(llm_chain.run(input_text))