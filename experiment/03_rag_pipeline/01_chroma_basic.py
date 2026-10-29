import chromadb
from llm_playbook.core.embeddings import GeminiEmbeddingFunction

chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

collection = chroma_client.get_or_create_collection(
    name="policies_gemini",
    embedding_function=GeminiEmbeddingFunction(),
)

file_path = "data/company_policy.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

raw_sections = content.split("## ")
documents = []
ids = []
metadatas = []

for idx, section in enumerate(raw_sections):
    clean_text = section.strip()
    if not clean_text or clean_text.startswith("# QUY CHẾ"):
        continue

    first_line = clean_text.split("\n")[0].strip()
    documents.append(clean_text)
    ids.append(f"chunk_{idx}")
    metadatas.append({"source": "company_policy.md", "section": first_line})

collection.upsert(
    documents=documents,
    ids=ids,
    metadatas=metadatas,
)

print(f"Đã nạp thành công {collection.count()} chunks vào ChromaDB với Gemini Vector!\n")

query = "Nhân viên thử việc có được hỗ trợ tiền gửi xe không?"
res = collection.query(query_texts=[query], n_results=1)

print(f"Câu hỏi: {query}")
print("Đoạn trích tìm thấy trong tài liệu:")
print(f"   ↳ {res['documents'][0][0]}")
print(f"Nguồn trích dẫn: {res['metadatas'][0][0]['section']}")
