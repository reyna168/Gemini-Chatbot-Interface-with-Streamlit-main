from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

from YiYan import YiYan

# 加载文件夹中的所有.md类型的文件
loader = DirectoryLoader('/work/langchain/data/', glob='**/*.md')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

# 初始化 embeddings 对象
embeddings = HuggingFaceEmbeddings(model_name='/shareData/text2vec-base-chinese')
# 将 document 计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
# 持久化数据
docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory="/work/langchain/Chromadb")
docsearch.persist()
# 加载数据
docsearch = Chroma(persist_directory="/work/langchain/Chromadb", embedding_function=embeddings)

# 创建问答对象
llm = YiYan()
prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用代码。
已知内容:
{context}
问题:
{question}"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": prompt}
qa = RetrievalQA.from_chain_type(llm=llm, retriever=docsearch.as_retriever(), chain_type="stuff",
                                 chain_type_kwargs=chain_type_kwargs, return_source_documents=True)

# 进行问答
query = "mongodb和redis的区别是什么"
res = qa({"query": query})
answer, docs = res['result'], res['source_documents']
print("\n\n> 问题:")
print(query)
print("\n> 回答:")
print(answer)
for document in docs:
    print("\n> " + document.metadata["source"] + ":")