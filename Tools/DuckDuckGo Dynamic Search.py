# DuckDuckGoSearchResults → If you want your application to search DuckDuckGo and fetch text snippets/URLs dynamically rather than loading a static web page.

import re
import os
from pprint import pp
import warnings
warnings.filterwarnings(action='ignore')

# Set a custom user-agent string
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

from langchain_core.documents.base import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader

search = DuckDuckGoSearchResults(max_results=1)
results = search.invoke("LangChain documentation") # returns a string → extract links using regular expressions

url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
links: list[str] = re.findall(url_pattern, results)

for link in links:
    pp(link)

loader = WebBaseLoader(web_paths=links)
docs: list[Document] = loader.load()

print(f"Loaded {len(docs)} documents from search results.")