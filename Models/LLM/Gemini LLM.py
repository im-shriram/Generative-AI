# Reference: https://reference.langchain.com/python/langchain-google-genai/llms/GoogleGenerativeAI
import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent.parent / ".env")

# Inherited from BaseLM
from langchain_google_genai import GoogleGenerativeAI 

llm = GoogleGenerativeAI(
    model="gemini-3-flash-preview", # You can not only use LLMs but also Chat Models
    google_api_key=os.getenv(key="GOOGLE_API_KEY")
)

response: str = llm.generate(
    prompts=["Who is the Prime Minister of India?"]
)

print(response) # The output is Generated String and no Meta-data