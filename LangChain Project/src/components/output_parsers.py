from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

class Schema(BaseModel):
    query_type: Literal["text", "code", "math", "unsecure", "all"] = Field(
        description="Classify the query to the following classes - 'text', 'code', 'math', 'unsecure', 'all'"
    )

class OutputParser:
    def __init__(self):
        pass

    def pydantic_parser(self) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=Schema)

    def string_parser(self) -> PydanticOutputParser:
        return StrOutputParser()