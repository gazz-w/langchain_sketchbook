import bs4
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

#  Retrieves documents from the web using the provided URL.


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=350
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

# create a vector database of documents


def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = faiss.FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

# Create a chain that retrieves the answer to the user's question


def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
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

    retriever = vectorStore.as_retriever(search_kwargs={"k": 2})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain


docs = get_documents_from_web(
    "https://pt.wikipedia.org/wiki/Oscar_2024")

vectorStore = create_db(docs)

chain = create_chain(vectorStore)


response = chain.invoke({
    "input": "what movie won the oscar in 2024?",

})
print(response)
