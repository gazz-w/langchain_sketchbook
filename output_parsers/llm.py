from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


# Instatiate Output Parser
parser = StrOutputParser()


# Instatiate Model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    verbose=False

)

# Prompt Template with messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "generate a list of 10 synonynms for the following word. Return the results as a comma separated list."),
    ("user", "{input}")
])


# Create a LLM Chain
chain = prompt | llm

response = chain.invoke({"input": "happy"})
print(type(response))  # return 'langchain_core.messeges.ai.AIMessage'
print(type(response.content))  # return 'str'


# PARSER
# result = parser.invoke(response)
# print(result)
