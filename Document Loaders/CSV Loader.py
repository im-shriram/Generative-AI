import pathlib
from langchain_core.documents.base import Document
from langchain_community.document_loaders import CSVLoader

csv_loader = CSVLoader(
    file_path=(pathlib.Path(__file__).parent / "data" / "social network.csv").absolute()
)
docs: list[Document] = csv_loader.load() # row-by-row documents

print(len(docs), docs[1], sep="\n\n")