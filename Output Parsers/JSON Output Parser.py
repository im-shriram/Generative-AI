import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

from pydantic import BaseModel, Field
from typing import Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages.ai import AIMessage
from langchain_core.prompt_values import PromptValue

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=1,
    max_tokens=2048
)

class Schema(BaseModel):
    """ A class inherited from BaseModel which is a defination of pydantic response schema the LLM is going to generate """

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg", "net"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    sentiment_probability: float = Field(gt=0.0, lt=1.0, default=0.5, description="Return a probability of positive sentement")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

# StructuredOutputParser is depricated, you can pass the schema directly inside JSON
output_parser = JsonOutputParser(pydantic_object=Schema) # The argument value must be a class and not a dictonary that defines json schema

template = ChatPromptTemplate(
    messages = [
        ("system", "You are an experienced {role}"),
        ("human", "Instead of giving me the response me in pure text (unstructured), you have to provide me the review in structured mannar. The review is - {review} \n{format_instructions}")
    ],
    input_variables=["role", "review"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}, # This variable "format_instruction" is filled at compile time → Returns: Return JSON object. Schema ....
    validate_template=True
)

prompt: PromptValue = template.invoke(
    input={
        "role": "Customer Service Agent and Business Analyst",
        "review": 
            """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

            The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

            However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

            Pros:
            Insanely powerful processor (great for gaming and productivity)
            Stunning 200MP camera with incredible zoom capabilities
            Long battery life with fast charging
            S-Pen support is unique and useful  

            Review by Dexter Morgan"""
    }
)

response: AIMessage = model.invoke(input=prompt)
result: str = output_parser.invoke(input=response)
print(result)