from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain

# system messages: instruct the LLM on how to act on human messages
# human messages: messages sent from the user
# AI-responses: responses from the AI

chat_llm = ChatOllama(model='mistral')

#instructions = SystemMessage(content="""
#                             You are a surfer dude, having a conversation about the surf conditions on the beach.
#                             Respond using surfer slang.
#                             """)
#
#question = HumanMessage(content="What is the weather like?")
#
#response = chat_llm.invoke([
#    instructions,
#    question
#    ])
#
#print(response.content)

# CREATING A CHAIN
#prompt = PromptTemplate(
#        template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
#        Respond using surfer slang.
#
#        Question: {question}
#        """,
#        input_variables=["question"]
#        )
#
#chat_chain = LLMChain(llm=chat_llm, prompt=prompt)
#
#response = chat_chain.invoke({"question": "What is the weather like?"})
#
#print(response)

# GROUNDING
prompt = PromptTemplate(
        template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
        Respond using surfer slang.

        Context: {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
        )

current_weather = """
    {
            "surf": [
                {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
                {"beach": "Polzeath", "conditions": "Flat and calm"},
                {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
                ]
    }"""

chat_chain = LLMChain(llm=chat_llm, prompt=prompt)

response = chat_chain.invoke(
        {
            "context": current_weather,
            "question": "What is the weather like?"
        }
    )

print(response)

