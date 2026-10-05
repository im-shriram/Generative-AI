from langchain_core.runnables import RunnablePassthrough

chain: RunnablePassthrough = RunnablePassthrough()
print(chain.invoke(
    input="Hello, How are you?"
))