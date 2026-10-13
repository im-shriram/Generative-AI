import pathlib
import warnings
from typing import Iterator
warnings.filterwarnings(action="ignore")

from langchain_core.documents.base import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

pdf_loader = DirectoryLoader(
    path=(pathlib.Path(__file__).parent / "data").absolute(),
    glob=["**/*.pdf", "*.pdf"],
    loader_cls=PyPDFLoader,
    show_progress=True,
)

txt_loader = DirectoryLoader(
    path=(pathlib.Path(__file__).parent / "data").absolute(),
    glob=["**/*.txt", "*.txt"],
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,
) # NOTE: DirectoryLoader cannot apply multiple loader at a same time, you have to create seperate dataloaders for each of them 

def lazy_loading():
    """ Load one document at a time (memory efficiency): Load → Chunk → Embed → Store """

    pdf_docs: Iterator[Document] = pdf_loader.lazy_load()
    txt_docs: Iterator[Document] = txt_loader.lazy_load()

    for document in pdf_docs:
        print(document.metadata)

    for document in txt_docs:
        print(document.metadata)

pdf_docs: list[Document] = pdf_loader.load()
txt_docs: list[Document] = txt_loader.load()

print( 
    f"Total Documents: {len(pdf_docs)}", 
    f"Page content: {pdf_docs[2].page_content}", 
    f"Page metadata: {pdf_docs[3].metadata}", 
    sep="\n\n"
)