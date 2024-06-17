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

# Prompt Template
prompt = ChatPromptTemplate.from_template("me conte uma piada de {subject}")

# Prompt Template with messages
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "you are an AI chef. Create a unique recipe base on the follow main ingridients"),
    ("user", "{input}")
])

# Create a LLM Chain
chain = prompt | llm

response = chain.invoke({"subject": "pintinho"})
# parser.invoke(response) is the same as response.content
result = parser.invoke(response)

print(response.content)
print("\n-----------------------------------\n")

# Create a LLM Chain to second prompt
chain2 = prompt2 | llm

response2 = chain2.invoke({"input": "rice"})
result2 = parser.invoke(response2)

print(result2)


# MODELO INVOKE
# response = llm.invoke("me conte uma piada de pintinho")

# MODELO DE BATCH
# response_batch = llm.batch(["Hello, how are you?", "how can you help me?"])

# print(response_batch)

# MODELO DE STREAMING
# response_stream = llm.stream("How can you help me?")

# for chunk in response_stream:
#     print(chunk.content, end="", flush=True)

# PARSER
# result = parser.invoke(response)
# print(result)
