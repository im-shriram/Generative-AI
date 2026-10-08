import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

from pydantic import BaseModel
from typing import Any

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
) # Only use models that are able to generate strunctured output

json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": [
                "pos",
                "neg"
            ],
            "description": "Return sentiment of the review either negative, positive or neutral"
        },
        "pros": {
            "type": [
                "array",
                "null"
            ],
            "items": {
                "type": "string"
            },
            "description": "Write down all the pros inside a list"
        },
        "cons": {
            "type": [
                "array",
                "null"
            ],
            "items": {
                "type": "string"
            },
            "description": "Write down all the cons inside a list"
        },
        "name": {
            "type": [
                "string",
                "null"
            ],
            "description": "Write the name of the reviewer"
        }
    },
    "required": [
        "key_themes",
        "summary",
        "sentiment"
    ]
}

structured_model: Runnable = model.with_structured_output(
    schema=json_schema, 
    method='json_schema', # "json_mode" is only of your prompt contains "json" string
    include_raw=True
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