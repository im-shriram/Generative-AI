import os
import pathlib
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent / ".env")

from langchain_groq import ChatGroq
from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSerializable

def load_docs(inputs: dict):
    loader: TextLoader = TextLoader(file_path=str(inputs['path']), encoding="utf-8")
    loader.invoke()
    docs: list[Document] = loader.load() # NOTE: Document Loader does not have "invoke" function → They are not "Runnables"
    return "\n".join(doc.page_content for doc in docs)

document_loader: RunnableLambda = RunnableLambda(func=load_docs)

template = PromptTemplate(
    template="Explain '{docs}' in a nutshell.",
    validate_template=True,
    input_variables=["docs"]
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_TOKEN"),
    reasoning_format="parsed",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

output_parser = StrOutputParser()

chain: RunnableSerializable = document_loader | template | model | output_parser
response: str = chain.invoke(input={
    'path': pathlib.Path(__file__).parent / "data" / "content.txt",
})
print(response)