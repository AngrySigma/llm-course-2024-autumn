from typing import List, Optional
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from part1.search_engine import Document, SearchResult


class FAISSSearcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализация индекса
        """
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = 384  # Размерность для 'all-MiniLM-L6-v2'

    def build_index(self, documents: List[Document]) -> None:
        """
        TODO: Реализовать создание FAISS индекса

        1. Сохранить документы
        2. Получить эмбеддинги через model.encode()
        3. Нормализовать векторы (faiss.normalize_L2)
        4. Создать индекс:
            - Создать quantizer = faiss.IndexFlatIP(dimension)
            - Создать индекс = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            - Обучить индекс (train)
            - Добавить векторы (add)
        """
        self.documents = documents
        embeddings = self.model.encode(
            [doc.text for doc in documents], convert_to_numpy=True
        )
        faiss.normalize_L2(embeddings)

        n_clusters = min(len(documents), 4)
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(
            quantizer, self.dimension, n_clusters, faiss.METRIC_INNER_PRODUCT
        )

        self.index.train(embeddings)
        self.index.add(embeddings)

    def save(self, path: str) -> None:
        """
        TODO: Реализовать сохранение индекса

        1. Сохранить в pickle:
            - documents
            - индекс (faiss.serialize_index)
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "index": faiss.serialize_index(self.index),
                },
                f,
            )

    def load(self, path: str) -> None:
        """
        TODO: Реализовать загрузку индекса

        1. Загрузить из pickle:
            - documents
            - индекс (faiss.deserialize_index)
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.index = faiss.deserialize_index(data["index"])

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Реализовать поиск

        1. Получить эмбеддинг запроса
        2. Нормализовать вектор
        3. Искать через index.search()
        4. Вернуть найденные документы
        """
        query_embedding = self.model.encode(query, convert_to_numpy=True).reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 indicates no result
                doc = self.documents[idx]
                results.append(
                    SearchResult(title=doc.title, text=doc.text, score=dist, doc_id=idx)
                )
        return results

    def batch_search(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[SearchResult]]:
        """
        TODO: Реализовать batch-поиск

        1. Получить эмбеддинги всех запросов
        2. Нормализовать векторы
        3. Искать через index.search()
        4. Вернуть результаты для каждого запроса
        """
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        faiss.normalize_L2(query_embeddings)

        distances, indices = self.index.search(query_embeddings, top_k)
        results = []
        for query_idx in range(len(queries)):
            query_results = []
            for dist, idx in zip(distances[query_idx], indices[query_idx]):
                if idx != -1:  # -1 indicates no result
                    doc = self.documents[idx]
                    query_results.append(
                        SearchResult(
                            title=doc.title, text=doc.text, score=dist, doc_id=idx
                        )
                    )
            results.append(query_results)
        return results
