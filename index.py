from github_embedding import GitHubEmbeddingMethod
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    
    # Embedding modelini oluştur
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
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
        repo="test-llm-repo-assistant",
        branch="main",
        github_token=os.environ.get("GITHUB_TOKEN"),
        ignore_directories=["node_modules", "dist", "tests"],
        ignore_file_extensions=[".png", ".jpg", ".md"]
    )

    try:
        # Pipeline'ı doğrudan burada oluştur ve embedding modelini ekle
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=20),
                embed_model  # Embedding modelini doğrudan ekle
            ],
            vector_store=vector_store,
        )

        print(f"[index_task_001] Dokümanlar yükleniyor...")
        documents = embedder.get_documents("test_repo")
        print(f"[index_task_001] {len(documents)} doküman yüklendi")

        documents = embedder.apply_rules(
            documents,
            inclusion_rules=[],  
            exclusion_rules=["test"]  
        )
        print(f"[index_task_001] {len(documents)} doküman filtreleme sonrası")

        print(f"[index_task_001] Düğümler oluşturuluyor ve vektör deposuna ekleniyor...")
        pipeline.run(documents=documents)
        
        print(f"[index_task_001] İndeksleme tamamlandı")
        print("Indexleme başarıyla tamamlandı!")
    except Exception as e:
        print(f"Indexleme hatası: {str(e)}")