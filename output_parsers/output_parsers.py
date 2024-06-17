from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
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


def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "extract information for the following phrase. \nFormatting Instructions: {format_instructions}"),
        ("user", "{phrase}")
    ])

    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")

    parser = JsonOutputParser(pydantic_object=Person)

    chain = prompt | model | parser

    return chain.invoke({
        "phrase": "Max is 30 year old",
        "format_instructions": parser.get_format_instructions()
    })


print(call_string_output_parser())

print(call_list_output_parser())

print(call_json_output_parser())

# if you want to confirm the type of each response, insert the "type()" function. eg.: print(type(call_string_output_parser()))
