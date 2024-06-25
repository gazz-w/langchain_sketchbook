from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
load_dotenv()

# create retriever

loader = WebBaseLoader('https://en.wikipedia.org/wiki/96th_Academy_Awards')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=350
)
splitDocs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectorStore = faiss.FAISS.from_documents(docs, embedding=embedding)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})


model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Max"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search = TavilySearchResults()
retriever_tool = create_retriever_tool(
    retriever,
    "oscar_2024_search",
    "use this tool when searchin for information about the 96th Academy Awards 2024"
)
tools = [search, retriever_tool]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)


def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history

    })

    return response["output"]


if __name__ == "__main__":

    chat_history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hello, how can I help you?"),
        HumanMessage(content="my name is Gabriel"),
    ]

    while True:
        user_input = input("you: ")
        if user_input.lower() == "exit":
            break

        response = process_chat(agentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant: ", response)
