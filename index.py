from github_embedding import GitHubEmbeddingMethod
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

# ChromaDB kurulumu
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("github_repos")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

class SimpleTaskManager:
    def init_task(self, task_id): 
        print(f"Task {task_id} started")
    def update_task(self, task_id, status): 
        print(f"Task {task_id}: {status}")

# Indexleme işlemi
embedder = GitHubEmbeddingMethod(
    owner="ZelihaBaysan",
    repo="ZelihaBaysan",
    branch="main",  # Branch artık load_data'da kullanılıyor
    github_token=os.environ.get("GITHUB_TOKEN"),
    ignore_directories=["node_modules", "dist", "tests"],
    ignore_file_extensions=[".png", ".jpg", ".md"]
)

try:
    embedder.process(
        vector_store=vector_store,
        task_manager=SimpleTaskManager(),
        data_source_id="github_repo",
        task_id="index_task_001",
        inclusion_rules=["src", "lib"],
        exclusion_rules=["test", "example"]
    )
    print("Indexleme başarıyla tamamlandı!")
except Exception as e:
    print(f"Indexleme hatası: {str(e)}")