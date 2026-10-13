import pathlib
import warnings
from pprint import pp
warnings.filterwarnings(action='ignore')

from langchain_core.documents.base import Document
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader(
    file_path=pathlib.Path(__file__).parent / "data" / "pdf" / "attention is all you need.pdf"
) # NOTE: This document loader is only used for extracting table content from pdfs, it removes all the space between words. For text content use PyPDFLoader

docs: list[Document] = loader.load()
pp(docs[5].page_content)