import bs4
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from dotenv import load_dotenv
load_dotenv()


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=350
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs


def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = faiss.FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.4,
        max_tokens=1000
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        # retriever,
        history_aware_retriever,
        chain
    )

    return retrieval_chain


def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history

    })
    return response["answer"]


if __name__ == "__main__":
    docs = get_documents_from_web(
        "https://pt.wikipedia.org/wiki/Oscar_2024")
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    chat_history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hello, how can I help you?"),
        HumanMessage(content="my name is John"),
    ]

    while True:
        user_input = input("you: ")
        if user_input.lower() == "exit":
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant: ", response)
