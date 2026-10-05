from langchain_core.runnables import RunnableLambda

def word_count(text: str) -> int:
    """ The output of previous runnable is passed in this function and accessed by the parameter you define """
    return len(text.split(sep=" "))

word_counter: RunnableLambda[str, int] = RunnableLambda(func=word_count)
print(word_counter.invoke(
    input="Hi, How are you?"
))