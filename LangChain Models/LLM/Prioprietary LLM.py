import os
import pathlib
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI # Inherited from BaseLM

load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

# Initialize the Gemini LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash", # No more available
    google_api_key=os.getenv(key="GOOGLE_API_KEY")
)

# Invoke the text completion model
response: str = llm.invoke("What is a Large Language Model?")
print(response)