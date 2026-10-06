from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import json
import pathlib
from typing import Any

from .components import Models, ChatHistory, CustomPromptTemplate, OutputParser
from .tasks import DocumentLoader, Embedding
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch

class Runnables:
    def __init__(
        self: Runnables,
        doc_loader_obj: DocumentLoader,
        embed_obj: Embedding,
        chat_prompt_template_obj: CustomPromptTemplate,
        models_obj: Models,
        output_parser_obj: OutputParser
    ):

        # Runnables for first chain
        self.doc_loader_obj = doc_loader_obj
        self.document_loader_runnable: RunnableLambda = RunnableLambda(
            func=self.loading_documents
        )
        self.text_splitter_runnable: RunnableLambda = RunnableLambda(
            func=self.splitting_texts
        )

        self.embed_obj = embed_obj
        self.embedding_document_runnable: RunnableLambda = RunnableLambda(
            func=self.embedding_docs
        )
        self.store_embeddings_runnable: RunnableLambda = RunnableLambda(
            func=self.store_embeddings
        )

        # Runnables for second chain
        self.embed_query: RunnableLambda = RunnableLambda(
            func=self.embedding_query
        )

        self.re_ranking: RunnableLambda = RunnableLambda(
            func=self.reranking
        )

        self.query_passthrough: RunnablePassthrough = RunnablePassthrough()

        self.chat_prompt_template_obj: CustomPromptTemplate = chat_prompt_template_obj
        self.system_prompt_template: ChatPromptTemplate = self.chat_prompt_template_obj.build_system_prompt_template()
        self.user_prompt_template: PromptTemplate = self.chat_prompt_template_obj.build_user_prompt_template()
        self.text_prompt_template = self.chat_prompt_template_obj.build_text_prompt_template()
        self.code_prompt_template = self.chat_prompt_template_obj.build_code_prompt_template()
        self.math_prompt_template = self.chat_prompt_template_obj.build_math_prompt_template()
        self.hybrid_prompt_template = self.chat_prompt_template_obj.build_hybrid_prompt_template()
        self.default_prompt_template = self.chat_prompt_template_obj.build_unsecure_prompt_template()

        self.models_obj: Models = models_obj
        self.chat_model: ChatGroq = self.models_obj.initialize_chat_model()

        self.output_parser_obj = output_parser_obj
        self.pydantic_parser: PydanticOutputParser[Any] = self.output_parser_obj.pydantic_parser()
        self.string_parser: PydanticOutputParser[Any] = self.output_parser_obj.string_parser()


    # Components for first chain
    def loading_documents(self, text: str="") -> str:
        return self.doc_loader_obj.load_documents(path=pathlib.Path(__file__).parent.parent / "data" / "knowledge base" / "knowledge_base.txt")

    def splitting_texts(self, doc: str) -> list[str]:
        return self.doc_loader_obj.split_documents(doc=doc)

    def embedding_docs(self, docs: list[str]) -> list[list[float]]:
        return self.embed_obj.embed_docs(docs=docs)

    def store_embeddings(self, embeddings: list[list[float]]) -> None:
        self.embed_obj.store_embeddings(embeddings=embeddings, path=pathlib.Path(__file__).parent.parent / "data" / "embeddings" / "knowledge_base_embeddings.json")
    
    # Components for second chain
    def embedding_query(self, query: str) -> list[float]:
        return self.embed_obj.embed_query(query=query["query"][-1].content)
    
    def reranking(self, embedded_query: list[float]) -> list[str]:
        with open(file=pathlib.Path(__file__).parent.parent / "data" / "embeddings" / "knowledge_base_embeddings.json", mode='r') as fp:
            embedded_knowledge_base = json.load(fp=fp)
        
        similarity_scores: list[tuple[int, float]] = []

        for idx, vector in enumerate(embedded_knowledge_base):
            similarity_scores.append((idx, sum([x * y for x, y in zip(embedded_query, vector)])))

        similarity_scores.sort(reverse=True, key=lambda a: a[1]) # sorting decending wrt similarity scores

        relevant_docs: list[str] = []
        for idx, score in similarity_scores:
            relevant_docs.append(self.splitting_texts(doc=self.loading_documents())[idx])

        return relevant_docs

class Pipeline:
    def __init__(
        self: Pipeline,
        runnable_obj: Runnables,
        chat_history_obj: ChatHistory
    ):
        self.runnable_obj: Runnables = runnable_obj
        self.chat_history_obj: ChatHistory = chat_history_obj

    def build_embedding_knowledge_base_chain(self) -> RunnableSerializable:
        chain: RunnableSerializable[Any, Any] = self.runnable_obj.document_loader_runnable | self.runnable_obj.text_splitter_runnable | self.runnable_obj.embedding_document_runnable | self.runnable_obj.store_embeddings_runnable
        return chain

    def build_response_chain(self) -> RunnableParallel:
        chain_1 = self.runnable_obj.embed_query | self.runnable_obj.re_ranking
        chain_2 = self.runnable_obj.user_prompt_template | self.runnable_obj.chat_model | self.runnable_obj.pydantic_parser

        parallel_chain = RunnableParallel({
            "relevent_docs": chain_1,
            "option": chain_2,
            "query": self.runnable_obj.query_passthrough
        })

        text_chain = self.runnable_obj.text_prompt_template | self.runnable_obj.chat_model | self.runnable_obj.string_parser
        code_chain = self.runnable_obj.code_prompt_template | self.runnable_obj.chat_model | self.runnable_obj.string_parser
        math_chain = self.runnable_obj.math_prompt_template | self.runnable_obj.chat_model | self.runnable_obj.string_parser
        hybrid_chain = self.runnable_obj.hybrid_prompt_template | self.runnable_obj.chat_model | self.runnable_obj.string_parser
        default_chain = self.runnable_obj.default_prompt_template | self.runnable_obj.chat_model | self.runnable_obj.string_parser

        conditional_chain = RunnableBranch(
            (lambda output_dict: output_dict["option"].query_type == "text", text_chain),
            (lambda output_dict: output_dict["option"].query_type == "code", code_chain),
            (lambda output_dict: output_dict["option"].query_type == "math", math_chain),
            (lambda output_dict: output_dict["option"].query_type == "all", hybrid_chain),
            RunnableLambda(func=lambda output_dict: default_chain)
        )

        return parallel_chain | conditional_chain

def main():

    runnable = Runnables(
        doc_loader_obj=DocumentLoader(),
        embed_obj=Embedding(model=Models().initialize_embedding_model()),
        chat_prompt_template_obj=CustomPromptTemplate(),
        models_obj=Models(),
        output_parser_obj=OutputParser()
    )
    chat_history_obj = ChatHistory()
    pipeline: Pipeline = Pipeline(runnable_obj=runnable, chat_history_obj=chat_history_obj)
    pipeline.build_response_chain().get_graph().print_ascii(); print("\n")
    return
    
    if False:
        pipeline.build_embedding_knowledge_base_chain().invoke(input="")

    current_conversation_history: list[str] = []
    system_prompt: PromptValue = runnable.system_prompt_template.invoke(
        input={"previous_chat_history": chat_history_obj.load_chat_history(path=pathlib.Path(__file__).parent.parent / "data" / "chat history" / "chat_history.json")}
    )
    current_conversation_history.extend(system_prompt.to_messages())

    while True:
        query: str = input("You: ")
        if query == "exit":
            break
        else:
            current_conversation_history.append(HumanMessage(content=query))
            response = pipeline.build_response_chain().invoke(
                input={"query": current_conversation_history}
            )
            print(f"AI: {response} \n\n")
            current_conversation_history.append(AIMessage(content=response))
    
    chat_history_obj.save_chat_history(
        chat_history=current_conversation_history,
        path=pathlib.Path(__file__).parent.parent / "data" / "chat history" / "chat_history.json"
    )

if __name__ == "__main__":
    main()