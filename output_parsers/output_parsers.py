from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000)


def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "tell me a joke about a follow subject"),
        ("user", "{subject}")
    ])

    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain.invoke({"subject": "dog"})


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "generate a list of 10 synonynms for the following word. Return the results as a comma separated list."),
        ("user", "{input}")
    ])

    parser = CommaSeparatedListOutputParser()

    chain = prompt | model | parser

    return chain.invoke({"input": "happy"})


print(call_string_output_parser())


print((call_list_output_parser()))

# this code: print(type(call_list_output_parser())) will return a 'list' type
