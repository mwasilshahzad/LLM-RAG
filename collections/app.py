# from langchain.vectorstores import Qdrant
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from qdrant_client import QdrantClient


# model_name = "BAAI/bge-large-en"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings" : False}


# embeddings = HuggingFaceBgeEmbeddings(
#     model_name = model_name,
#     model_kwargs = model_kwargs,
#     encode_kwargs = encode_kwargs
# )

# url = "http://localhost:6333"
# collection_name = "gpt_db"

# client = QdrantClient(
#     url = url,
#     prefer_grpc = False
# )

# print(client)
# print("--------------------------------")

# db = Qdrant(
#     client = client,
#     embeddings = embeddings,
#     collection_name = collection_name
# )

# print(db)
# print("---------------------------------")

# query = "What are classical approaches to tree detection problem ?"

# # On the basis of Similarity
# docs = db.similarity_search_with_score(query=query, k=5)

# for i in docs:
#     doc, score = i
#     print({"score": score, "content": doc.page_content, "metadata": doc.metadata})


from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from transformers import BertForConditionalGeneration, BertTokenizer
import torch

# Load GPT-2 model and tokenizer for text generation
generation_model_name = "gpt2"
generation_model = GPT2LMHeadModel.from_pretrained(generation_model_name)
generation_tokenizer = GPT2Tokenizer.from_pretrained(generation_model_name)

# Specify device for generation model
generation_device = "cuda" if torch.cuda.is_available() else "cpu"
generation_model.to(generation_device)

# Set up Qdrant and embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"
collection_name = "gpt_db"

client = QdrantClient(
    url=url,
    prefer_grpc=False
)

print(client)
print("--------------------------------")

db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name=collection_name
)

print(db)
print("---------------------------------")

# Query for similar documents
query = "What are classical approaches to tree detection problem?"
docs = db.similarity_search_with_score(query=query, k=5)

# Generate text using GPT-2 model for each retrieved document
for i in docs:
    doc, score = i
    prompt = f"Similarity score: {score:.4f}. Document: {doc.page_content}."

    # Tokenize and generate text
    input_ids = generation_tokenizer.encode(prompt, return_tensors="pt").to(generation_device)
    output_ids = generation_model.generate(input_ids, max_length=200, num_beams=5, no_repeat_ngram_size=2)
    
    # Manually truncate the output sequence to the specified max_length
    output_ids = output_ids[:, :100]  # Adjust the length as needed

    generated_text = generation_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print({"score": score, "content": generated_text, "metadata": doc.metadata})