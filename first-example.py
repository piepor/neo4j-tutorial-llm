from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

with open('./.openai-key', 'r') as file:
    openai_api_key = file.read()

llm = Ollama(model="mistral", temperature=0.5)

template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables=["fruit"])

response = llm.invoke(template.format(fruit='apple'))

print(response)
