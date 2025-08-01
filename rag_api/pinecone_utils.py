import os, pinecone

api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENV")
pinecone.init(api_key=api_key, environment=environment)
index_name = os.getenv("PINECONE_INDEX","vision-rag")
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric="cosine")
index = pinecone.Index(index_name)