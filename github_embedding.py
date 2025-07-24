import multiprocessing
from typing import List, Sequence, Optional
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.readers.github import GithubRepositoryReader, GithubClient


class VectorStoreProtocol:
    def add(self, nodes: Sequence[BaseNode]) -> None:
        pass


class TaskManagerProtocol:
    def init_task(self, task_id: str) -> None:
        pass

    def update_task(self, task_id: str, status: str) -> None:
        pass


class TaskStatus:
    DONE = "DONE"
    ERROR = "ERROR"


class GitHubEmbeddingMethod:
    def __init__(
        self,
        owner: str,
        repo: str,
        branch: Optional[str] = "main",
        use_parser: bool = True,
        verbose: bool = True,
        github_token: Optional[str] = None,
        ignore_directories: Optional[List[str]] = None,
        ignore_file_extensions: Optional[List[str]] = None,
    ):
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.use_parser = use_parser
        self.verbose = verbose
        self.github_token = github_token
        self.ignore_directories = ignore_directories or []
        self.ignore_file_extensions = ignore_file_extensions or []

    @staticmethod
    def customize_metadata(document: Document, data_source_id: str) -> Document:
        document.metadata = {
            "file_path": document.metadata.get("file_path", ""),
            "file_name": document.metadata.get("file_name", ""),
            "data_source_id": data_source_id,
        }
        return document

    def apply_rules(
        self,
        documents: Sequence[Document],
        inclusion_rules: List[str],
        exclusion_rules: List[str],
    ) -> Sequence[Document]:
        filtered_docs = []
        for doc in documents:
            file_path = doc.metadata.get("file_path", "").lower()
            file_name = doc.metadata.get("file_name", "").lower()

            # Exclusion kontrolü
            if any(excl.lower() in file_path or excl.lower() in file_name for excl in exclusion_rules):
                continue

            # Inclusion kontrolü (eğer inclusion_rules boşsa tümünü kabul et)
            if not inclusion_rules or any(incl.lower() in file_path or incl.lower() in file_name for incl in inclusion_rules):
                filtered_docs.append(doc)

        print(f"Filtreleme detayı: {len(documents)} -> {len(filtered_docs)} doküman")
        return filtered_docs

    def get_documents(self, data_source_id: str) -> List[Document]:
        github_client = GithubClient(
            github_token=self.github_token,
            verbose=self.verbose
        )

        loader = GithubRepositoryReader(
            github_client=github_client,
            owner=self.owner,
            repo=self.repo,
            use_parser=self.use_parser,
            verbose=self.verbose,
            filter_directories=(
                self.ignore_directories,
                GithubRepositoryReader.FilterType.EXCLUDE
            ),
            filter_file_extensions=(
                self.ignore_file_extensions,
                GithubRepositoryReader.FilterType.EXCLUDE
            ),
        )

        documents = loader.load_data(branch=self.branch)
        for document in documents:
            self.customize_metadata(document, data_source_id)
        return documents

    def get_nodes(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=20)
            ]
        )
        num_workers = multiprocessing.cpu_count()
        return pipeline.run(documents=documents, num_workers=num_workers)

    def process(
        self,
        vector_store: VectorStoreProtocol,
        task_manager: TaskManagerProtocol,
        data_source_id: str,
        task_id: str,
        **kwargs,
    ) -> None:
        task_manager.init_task(task_id)
        try:
            print(f"[{task_id}] Dokümanlar yükleniyor...")
            documents = self.get_documents(data_source_id)
            print(f"[{task_id}] {len(documents)} doküman yüklendi")

            documents = self.apply_rules(
                documents,
                inclusion_rules=kwargs.get("inclusion_rules", []),
                exclusion_rules=kwargs.get("exclusion_rules", []),
            )
            print(f"[{task_id}] {len(documents)} doküman filtreleme sonrası")

            print(f"[{task_id}] Düğümler oluşturuluyor...")
            nodes = self.get_nodes(documents)
            print(f"[{task_id}] {len(nodes)} düğüm oluşturuldu")

            print(f"[{task_id}] Vektör deposuna ekleniyor...")
            vector_store.add(nodes)
            task_manager.update_task(task_id, TaskStatus.DONE)
            print(f"[{task_id}] İndeksleme tamamlandı")
        except Exception as e:
            print(f"[{task_id}] Hata: {str(e)}")
            task_manager.update_task(task_id, TaskStatus.ERROR)
            raise
