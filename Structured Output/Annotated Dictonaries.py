import os
from dotenv import load_dotenv
import pathlib
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

from pydantic.main import BaseModel
from typing import (
    TypedDict, # dictonary with defined schema 
    Annotated, # annotations along with type
    Optional, # define a perticular value could be optional
    Literal, # defining set of values
    Any
)

from langchain_core.prompt_values import PromptValue
from langchain_core.runnables.base import Runnable
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_TOKEN"),
    reasoning_format="parsed",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# NOTE: TypedDict schema is completely optional. If you store a value having a datatype other than specified in schema then python neither throw an error nor give us a warning rather it allows to store that literal - No validation present.
class Review(TypedDict):
    """ A class inherited from TypedDict which is a defination of response schema the LLM is going to generate """

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    

structured_model: Runnable = model.with_structured_output(
    schema=Review,
    method="function_calling", # tool-call
    include_raw=True # The final output is always a dict with keys 'raw', 'parsed', and 'parsing_error'.
) 

template = ChatPromptTemplate(
    messages = [
        ("system", "You are an experienced {role}"),
        ("human", "Instead of giving me the response me in pure text (unstructured), you have to provide me the review in structured mannar. The review is {review}")
    ],
    input_variables=["role", "review"],
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

result: BaseModel | dict[Any, Any] = structured_model.invoke(
    input=prompt
)

print(result["raw"], result["parsed"], result["parsing_error"], sep="\n\n")