import pathlib

class DocumentLoader:
    def __init__(self) -> None:
        self.data: str = ""
        self.knowledge_base: list[str] = [""]

    def load_documents(self: DocumentLoader, path: pathlib.Path) -> str:
        with open(file=path, mode="r") as fp:
            self.data = fp.read()
            return self.data
    
    def split_documents(self: DocumentLoader, doc: str) -> list[str]:
        self.knowledge_base = doc.split(sep=". ")
        return self.knowledge_base

def main():
    path: pathlib.Path = pathlib.Path(__file__).parent.parent / "data" / "knowledge_base.txt"

    document_loader = DocumentLoader()
    data, knowledge_base = document_loader.load_documents(path=path)
    
    print(f"Data: {data} \n")
    print(f"Splitted Data: {knowledge_base}")

if __name__ == "__main__":
    main()