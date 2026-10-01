from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
import pathlib
from dotenv import load_dotenv

load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

response: str = model.invoke("Hi there, what is your name and how are you doing?")
print(response)