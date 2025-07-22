from github_embedding import GitHubEmbeddingMethod
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# ChromaDB kurulumu
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("github_repos")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Basit görev yöneticisi
class SimpleTaskManager:
    def init_task(self, task_id): print(f"Task {task_id} started")
    def update_task(self, task_id, status): print(f"Task {task_id}: {status}")

# Indexleme işlemi
embedder = GitHubEmbeddingMethod(
    owner="ZelihaBaysan",
    repo="ZelihaBaysan",
    github_token="",
    ignore_directories=["node_modules", "dist", "tests"],
    ignore_file_extensions=[".png", ".jpg", ".md"]
)

embedder.process(
    vector_store=vector_store,
    task_manager=SimpleTaskManager(),
    data_source_id="openai_python",
    task_id="task_123",
    inclusion_rules=["openai", "api"],
    exclusion_rules=["test", "example"]
)
