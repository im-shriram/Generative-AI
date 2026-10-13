import warnings
from typing import Iterator
warnings.filterwarnings(action='ignore')

from langchain_core.documents.base import Document
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader # throwing error → create custom document loader class with selenium operations

url: list[str] = [
    "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421",
    "https://en.wikipedia.org/wiki/Transformers",
    "https://arxiv.org/abs/1706.03762"
]
    
loader = WebBaseLoader(web_path=url) # one document per website
docs: Iterator[Document] = loader.lazy_load()

for doc in docs:
    print(doc, end="\n\n")