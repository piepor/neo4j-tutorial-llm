from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

chat_llm = ChatOllama(model='mistral')

prompt = PromptTemplate(template="""
                        You are a surfer dude, having a conversation about the surf conditions on the beach.
                        Respond using surfer slang.

                        Chat History: {chat_history}
                        Context: {context}
                        Question: {question}
                        """, input_variables=["chat_history", "context", "question"])

current_weather = """
    {
            "surf": [
                {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
                {"beach": "Polzeath", "conditions": "Flat and calm"},
                {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
                ]
    }"""

memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

chat_chain = LLMChain(llm=chat_llm, prompt=prompt, memory=memory)

#response = chat_chain.invoke({
#    "context": current_weather,
#    "question": "Hi, I am Watergate Bay. What is the surf like?"
#    })
#
#print(response["text"])
#
#response = chat_chain.invoke({
#    "context": current_weather,
#    "question": "Where I am?"
#    })
#
#print(response["text"])

while True:
    question = input("> ")
    response = chat_chain.invoke({
        "context": current_weather,
        "question": question
        })

    print(response['text'])
