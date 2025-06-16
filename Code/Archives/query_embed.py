import chromadb
from sentence_transformers import SentenceTransformer

# === CONFIG ===
DB_DIR = "C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/ChromaDB"
COLLECTION_NAME = "repair_embeddings"

# === INITIALIZE CHROMADB ===
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

# === LOAD EMBEDDING MODEL ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === USER QUERY ===
query = input("🔍 Enter a repair concern (e.g., 'engine knocking when accelerating'): ")
embedding = model.encode([query])[0].tolist()

# === SEARCH IN CHROMADB ===
results = collection.query(
    query_embeddings=[embedding],
    n_results=5
)

# === DISPLAY RESULTS ===
print("\nTop 5 Similar Past Repairs:\n")
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print("---")
    print(doc)
    print("🔧 Job Type:", meta.get("job_type"))
    print("🧩 Part:", meta.get("part_name"))
    print("🚗 Vehicle:", f"{meta.get('make')} {meta.get('model')} {meta.get('year')}")
print("\n✅ Retrieval complete.")
