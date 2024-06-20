import bs4
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from dotenv import load_dotenv
load_dotenv()


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs


def create_vector(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = faiss.FAISS.from_documents(docs, embedding=embedding)


docs = get_documents_from_web(
    "https://pt.wikipedia.org/wiki/Oscar_2024")

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.4,
    max_tokens=1000
)

prompt = ChatPromptTemplate.from_template("""
Answer the user's question
Context: {context}
Question: {input}                                        
""")

chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,

)


response = chain.invoke({
    "input": "what movie won the oscar in 2024?",
    "context": docs
})
print(response)
