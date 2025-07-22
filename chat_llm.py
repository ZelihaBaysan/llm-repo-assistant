from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Embedding model ayarı
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Gemma modelini başlat
Settings.llm = Ollama(model="gemma:7b", request_timeout=60.0)

# ChromaDB vektör veritabanını yükle
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("github_repos")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Index'i oluştur
index = VectorStoreIndex.from_vector_store(vector_store)

# Sohbet motorunu başlat
query_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    verbose=True,
    system_prompt="You're an expert on the given GitHub repository. Provide detailed answers about the codebase."
)

# Kullanıcı etkileşimi
print("GitHub Repo Sohbet Asistanı (Gemma 7B) - Çıkmak için 'exit' yazın")
while True:
    query = input("\nKullanıcı: ")
    if query.lower() == "exit":
        break
        
    response = query_engine.chat(query)
    print(f"\nAsistan: {response}")