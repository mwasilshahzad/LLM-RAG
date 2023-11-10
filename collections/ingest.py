from langchain.vectorstores import Qdrant
# Well performing embedding model
from langchain.embeddings import HuggingFaceBgeEmbeddings
# To handle PDF to extract text 
from langchain.document_loaders import PyPDFLoader
# Token | Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
texts = text_splitter.split_documents(documents)

# Load the embedding model

model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
# Normlization of embeddings
encode_kwargs = {"normalize_embeddings" : False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

print("Embedding Model Loaded.........")

# Qdrant URL
url = "http://localhost:6333"
# Collection name or DB collection name
collection_name = "gpt_db"

# Initlizations
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    # Interact Microservices, scalable distribution system, protobuf grpc
    prefer_grpc = False,
    collection_name = collection_name
)

print("Qdrant Index Created...........")