from langchain_core.prompt_values import PromptValue
import pydantic.main
from langchain_core.runnables.base import Runnable
from langchain_groq.chat_models import LanguageModelInput
from pydantic import BaseModel, Field
from typing import Any
import os
from dotenv import load_dotenv
import pathlib
from typing import Optional, Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Loading environment variables
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")

# Defininf Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_TOKEN"),
    reasoning_format="parsed",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class Review(BaseModel):
    """
        PyDantic schema is completely required. If you store a value having a datatype other than specified in schema then python throw an error - No validation present.
    """

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg", "net"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    sentiment_probability: float = Field(gt=0.0, lt=1.0, default=0.5, description="Return a probability of positive sentement")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    

structured_model: Runnable[LanguageModelInput, pydantic.main.BaseModel | dict[Any, Any]] = model.with_structured_output(schema=Review, include_raw=True)

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
print("Result:", result, "\n\n")

if result["parsing_error"] is not None:
    print("Validation failed:", result["parsing_error"])
    print("Raw output was:", result["raw"])
else:
    validated = result["parsed"] # Pydantic instance
    print("Pydantic Response:", validated)