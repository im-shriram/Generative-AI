from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.runnables import RunnableSerializable
import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")

from pydantic import BaseModel, Field
from typing import Literal

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

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

model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5,
    max_output_tokens=2048,
    disable_streaming=False
)

sequential_chain: RunnableAssign = RunnablePassthrough.assign(
    sentiment = sentiment_template | model | sentiment_output_parser
)
conditional_chain: RunnableBranch = RunnableBranch(
    (lambda review: review["sentiment"].sentiment == "positive", positive_response_template | model | response_output_parser),
    (lambda review: review["sentiment"].sentiment == "negative", negative_response_template | model | response_output_parser),
    RunnableLambda(lambda review: "The sentiment of review is neutral")
)
system_chain: RunnableSerializable = sequential_chain | conditional_chain
response = system_chain.invoke(
    input={
        "review": """
            I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
    
            The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
    
            However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
        """
    }
)
print(response)