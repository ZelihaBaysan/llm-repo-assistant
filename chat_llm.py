from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.memory import ChatMemoryBuffer

# Gelişmiş ayarlar
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="gemma:7b", request_timeout=120.0)
Settings.chunk_size = 512
Settings.chunk_overlap = 50

def initialize_chat_engine():
    try:
        # ChromaDB bağlantısı
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("github_repos")
        
        # Vektör deposunu kontrol et
        if chroma_collection.count() == 0:
            raise ValueError("Vektör deposu boş! Lütfen önce index.py ile verileri indeksleyin.")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Bellek (geçmiş konuşmalar) ayarı
        memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        
        # Index oluştur
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=Settings.embed_model
        )
        
        # Sohbet motorunu oluştur
        query_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            verbose=True,
            memory=memory,
            system_prompt=(
                "Sen bir GitHub repository uzmanısın. "
                "Kullanıcıların repository ile ilgili teknik sorularını detaylı ve açıklayıcı şekilde yanıtlıyorsun. "
                "Eğer bir konuda emin değilsen, 'Bilmiyorum' demekten çekinme."
            )
        )
        return query_engine
        
    except Exception as e:
        print(f"Başlatma hatası: {str(e)}")
        return None

def main():
    print("\nGitHub Repo Sohbet Asistanı (Gemma 7B) - Çıkmak için 'exit' yazın")
    
    query_engine = initialize_chat_engine()
    if not query_engine:
        return
    
    try:
        while True:
            query = input("\nKullanıcı: ").strip()
            if query.lower() in ['exit', 'quit', 'çık']:
                break
                
            if not query:
                print("Lütfen geçerli bir soru girin.")
                continue
                
            try:
                response = query_engine.chat(query)
                print(f"\nAsistan: {response}")
            except Exception as e:
                print(f"\nSoru işlenirken hata oluştu: {str(e)}")
                
    except KeyboardInterrupt:
        print("\nProgram sonlandırılıyor...")
    finally:
        print("Görüşmek üzere!")

if __name__ == "__main__":
    main()