import fsspec
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsRedundantFilter,
    LongContextReorder
)
from langchain.retrievers.merger_retriever import MergerRetriever
from dotenv import load_dotenv
load_dotenv()

class DataProcessor:
    def __init__(self) -> None:
        self.fs = fsspec.filesystem(
            "s3",
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            client_kwargs={"endpoint_url": os.getenv("ENDPOINT_URL")},
            config_kwargs={"s3": {"addressing_style": "path"}} 
        )
        self.mlflow_logging_url = f"{os.getenv('ENDPOINT_URL')}/mlflow/"
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.base_db_storage = "../data/02_output/faiss_ollama"

    def fetch_data(self, file_name: str) -> pd.DataFrame:
        file_path = f"datafiles/{file_name}"
        with self.fs.open(file_path, "rb") as f:
            if str(file_path).endswith(".csv"):
                print("It's a csv!")
                return pd.read_csv(f)
            elif str(file_path).endswith(".json"):
                print("It's a json!")
                return pd.read_json(f)
            elif str(file_path).endswith(".parquet"):
                print("It's a parquet!")
                return pd.read_parquet(f)
            else:
                raise ValueError(f"Unsupported file: {file_path}")
            
    
    def upload_data(self, file_path: str) -> None:
        self.fs.put(file_path, "datafiles/")
        print("Data successfully loaded")

    def delete_data(self, file_name: str) -> None:
        self.fs.rm(f"datafiles/{file_name}")
        print("Data successfully deleted")


    def db_encoding(self, data_file):
        raw_texts = self.fetch_data(data_file)
        documents = [
            Document(page_content=text, metadata={"id": f"row_{i}"})
            for i, text in enumerate(raw_texts)
        ]
        return documents

    def create_retriever(self, data_file: str):
        ollama_embeddings = OllamaEmbeddings(base_url=self.ollama_host, model="mxbai-embed-large")

        if os.path.exists(self.base_db_storage) and os.path.isdir(self.base_db_storage):
            faiss_index = FAISS.load_local(self.base_db_storage, ollama_embeddings, allow_dangerous_deserialization=True)
        else:
            documents = self.db_encoding(data_file)
            os.makedirs(self.base_db_storage, exist_ok=True)
            faiss_index = FAISS.from_documents(documents, embedding=ollama_embeddings)
            faiss_index.save_local(self.base_db_storage)

        retriever = faiss_index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        print("Retrievercreated successfully.")
        return retriever

    def create_complex_retriever(self, data_file: str):
        documents = self.db_encoding(data_file)

        emb_all = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        emb_mq  = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-dot-v1")

        FAISS.from_documents(documents, emb_all).save_local("faiss_all")
        FAISS.from_documents(documents, emb_mq).save_local("faiss_mq")

        ret_all = FAISS.load_local("faiss_all", emb_all, allow_dangerous_deserialization=True)\
                    .as_retriever(search_type="similarity", search_kwargs={"k": 5})
        ret_mq  = FAISS.load_local("faiss_mq", emb_mq, allow_dangerous_deserialization=True)\
                    .as_retriever(search_type="mmr", search_kwargs={"k": 5})

        lotr = MergerRetriever(retrievers=[ret_all, ret_mq])

        filter_embeddings = OllamaEmbeddings(base_url=self.ollama_host, model="mxbai-embed-large")
        filter = EmbeddingsRedundantFilter(embeddings=filter_embeddings)
        reordering = LongContextReorder()
        pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])

        compression_lotr_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline,
            base_retriever=lotr
        )

        print("Retrievers and compression pipeline created successfully.")
        return compression_lotr_retriever
