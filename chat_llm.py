from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.memory import ChatMemoryBuffer
import logging
from settings import initialize_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ayarları başlat
initialize_settings()

def initialize_chat_engine():
    try:
        # ChromaDB bağlantısı
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_collection("github_repos")
        
        count = chroma_collection.count()
        logger.info(f"Vektör deposunda {count} doküman bulundu")
        
        if count == 0:
            raise ValueError("Vektör deposu boş! Lütfen önce index.py'yi çalıştırın.")
        
        # Index oluştur
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Sohbet motoru
        return index.as_chat_engine(
            chat_mode="condense_plus_context",
            verbose=True,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
            system_prompt="Sen bir GitHub repository uzmanısın. Teknik soruları detaylıca yanıtlıyorsun."
        )
        
    except Exception as e:
        logger.error(f"Başlatma hatası: {str(e)}")
        return None

def main():
    logger.info("\n=== GitHub Repo Sohbet Asistanı ===")
    logger.info("Çıkmak için 'exit' yazın\n")
    
    query_engine = initialize_chat_engine()
    if not query_engine:
        return
    
    try:
        while True:
            query = input("Kullanıcı: ").strip()
            if query.lower() in ['exit', 'quit', 'çık']:
                break
                
            try:
                response = query_engine.chat(query)
                print(f"\nAsistan: {response}\n")
            except Exception as e:
                logger.error(f"Soru işlenirken hata: {str(e)}")
                
    except KeyboardInterrupt:
        logger.info("\nProgram sonlandırılıyor...")
    finally:
        logger.info("Görüşmek üzere!")

if __name__ == "__main__":
    main()