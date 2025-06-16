
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

# === CONFIG ===
CHROMA_PATH = r"C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/chroma_repair_orders"
COLLECTION_NAME = "repair_orders"
EMBEDDING_MODEL = "text-embedding-3-small"

# === Load Persisted Vector Store ===
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

# === Get User Query ===
query = input("🔍 Enter a repair concern (e.g., 'engine knocking when accelerating'): ")

# === Search Top 5 Similar Docs ===
results = vectorstore.similarity_search_with_score(query, k=5)

# === Display Results ===
print("\nTop 5 Similar Repair Orders:\n")
if not results:
    print("❌ No matches found.")
else:
    for doc, score in results:
        print("---")
        print(doc.page_content[:1000])  # Limit output
        print("📄 Metadata:", doc.metadata)
        print(f"🔍 Score: {score:.2f}")
print("\n✅ Retrieval complete.")