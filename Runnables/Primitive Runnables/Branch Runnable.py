from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.runnables import RunnableSerializable
import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent.parent / ".env")

from pydantic import BaseModel, Field
from typing import Literal

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableSequence

class Schema(BaseModel):
    """ A pydantic schema class through which LLMs can generate the structured output """
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Return sentiment of the review either negative, positive or neutral")
sentiment_output_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Schema)
response_output_parser: StrOutputParser = StrOutputParser()

sentiment_template: PromptTemplate = PromptTemplate(
    template="""
        Product Review - {review}
        Output Schema - {output_format}
        Based on the provided product review given by a customer, your task is to generate a response in the provided output schema
    """,
    input_variables=["review"],
    partial_variables={
        "output_format": sentiment_output_parser.get_format_instructions()
    },
    validate_template=True
)
positive_response_template: PromptTemplate = PromptTemplate(
    template="""
        Product Review: {review}
        Sentiment: {sentiment}
        Write a short, context-aware reply to this customer review for the positive sentiment. Keep it professional, specific to what they mentioned, and 2-3 sentences max.
""",
    input_variables=["review", "sentiment"],
    validate_template=True
)
negative_response_template: PromptTemplate = PromptTemplate(
    template="""
        Product Review: {review}
        Sentiment: {sentiment}
        Write a short, context-aware reply to this customer review for the negative sentiment. Keep it professional, specific to what they mentioned, and 2-3 sentences max.
""",
    input_variables=["review", "sentiment"],
    validate_template=True
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=1.5, # for creativity
    api_key=os.getenv("GROQ_API_TOKEN")
)
"""
    model_without_automatic_function_calling = model.bind(
        automatic_function_calling={"disable": True}
    )
"""
def preprocess(text: dict[str, str]):
    # Using this syntex because text [input] could have other information as well
    return {**text, "review": text["review"].lower()}
preprocess_runnable = RunnableLambda(func=preprocess)
parallel_chain = RunnableParallel({
    "review":  RunnablePassthrough(), # output "review"
    "sentiment": RunnableSequence(preprocess_runnable, sentiment_template, model, sentiment_output_parser) # output "sentiment"
}) # In Chains/Conditional Chains → Used RunnablePassthrough.assign(), but the above one is simpler
conditional_chain: RunnableBranch = RunnableBranch(
    (lambda review: review["sentiment"].sentiment == "positive", positive_response_template | model | response_output_parser),
    (lambda review: review["sentiment"].sentiment == "negative", negative_response_template | model | response_output_parser),
    RunnableLambda(lambda review: "The sentiment of provided review is neutral")
)

def report_sentiment(review_result):
    sentiment = review_result["sentiment"].sentiment
    print(f"Detected {sentiment} sentiment. Drafting response:", flush=True)
    return review_result

system_chain: RunnableSerializable = parallel_chain | RunnableLambda(report_sentiment) | conditional_chain
review_input = {
    "review": """
            I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
    
            The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
    
            However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
        """
}

print("Classifying review...", flush=True)
print(system_chain.invoke(
    input=review_input
), end="\n\n")
system_chain.get_graph().print_ascii(); print("\n")