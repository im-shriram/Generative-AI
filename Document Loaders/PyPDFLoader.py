import os
import pathlib
import warnings
from pprint import pp
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
warnings.filterwarnings(action='ignore')

from langchain_groq import ChatGroq
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader # "langchain_community" is going to get deprecated
from langchain_community.document_loaders.parsers import LLMImageBlobParser

document_loader = PyPDFLoader(
    file_path=pathlib.Path(__file__).parent / "data" / "notes.pdf", 
    password=None,  
    mode='page',
    extract_images=True,

    # Use a vision-capable model to read and describe extracted images.
    images_parser=LLMImageBlobParser(
        model=ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=os.getenv(key="GROQ_API_TOKEN"))
    ),
    images_inner_format="markdown-img" # Formats how the image output gets injected into page content
)

docs: list[Document] = document_loader.load() # Returns a "Document Object" → page_content - string and metadata - dictonary
print(docs)