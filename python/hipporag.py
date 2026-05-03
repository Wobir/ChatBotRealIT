"""Модуль для работы с HippoRAG2 RAG системой."""
import os
from typing import List, Optional, Dict, Any
from hipporag import HippoRAG


class HippoRAGService:
    """Сервис для интеграции HippoRAG2 в проект."""

    def __init__(
        self,
        save_dir: str = "outputs",
        llm_model_name: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        embedding_model_name: str = "nvidia/NV-Embed-v2",
        embedding_base_url: Optional[str] = None,
    ):
        """
        Инициализация HippoRAG сервиса.

        Args:
            save_dir: Директория для сохранения объектов HippoRAG.
            llm_model_name: Название модели LLM (например, 'gpt-4o-mini' или локальная модель).
            llm_base_url: URL для LLM (для локальных vLLM серверов, например 'http://localhost:8000/v1').
            embedding_model_name: Название модели эмбеддингов.
            embedding_base_url: URL для embedding endpoint (опционально).
        """
        self.save_dir = save_dir
        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.embedding_model_name = embedding_model_name
        self.embedding_base_url = embedding_base_url

        self.hipporag: Optional[HippoRAG] = None
        self._initialized = False

    def initialize(self):
        """Инициализация HippoRAG экземпляра."""
        if self._initialized:
            return

        kwargs = {
            "save_dir": self.save_dir,
            "embedding_model_name": self.embedding_model_name,
        }

        if self.llm_model_name:
            kwargs["llm_model_name"] = self.llm_model_name

        if self.llm_base_url:
            kwargs["llm_base_url"] = self.llm_base_url

        if self.embedding_base_url:
            kwargs["embedding_base_url"] = self.embedding_base_url

        self.hipporag = HippoRAG(**kwargs)
        self._initialized = True
        print(f"HippoRAG инициализирован с save_dir={self.save_dir}")

    def index_documents(self, docs: List[str]):
        """
        Индексация документов в HippoRAG.

        Args:
            docs: Список текстов документов для индексации.
        """
        if not self._initialized:
            self.initialize()

        self.hipporag.index(docs=docs)
        print(f"Проиндексировано {len(docs)} документов")

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов для запросов.

        Args:
            queries: Список поисковых запросов.
            num_to_retrieve: Количество документов для возврата на каждый запрос.

        Returns:
            Список результатов поиска.
        """
        if not self._initialized:
            self.initialize()

        return self.hipporag.retrieve(queries=queries, num_to_retrieve=num_to_retrieve)

    def rag_qa(
        self,
        queries: List[str],
        gold_docs: Optional[List[List[str]]] = None,
        gold_answers: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ответы на вопросы с использованием RAG.

        Args:
            queries: Список вопросов.
            gold_docs: Опционально. Золотые документы для оценки.
            gold_answers: Опционально. Золотые ответы для оценки.

        Returns:
            Список ответов на вопросы.
        """
        if not self._initialized:
            self.initialize()

        return self.hipporag.rag_qa(
            queries=queries,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
        )

    def rag_qa_with_context(
        self,
        query: str,
        context_messages: List[Dict[str, str]],
        num_to_retrieve: int = 3,
    ) -> str:
        """
        Ответ на вопрос с использованием контекста диалога и RAG.

        Args:
            query: Текущий вопрос пользователя.
            context_messages: История диалога для понимания контекста.
            num_to_retrieve: Количество документов для поиска.

        Returns:
            Текстовый ответ на вопрос.
        """
        if not self._initialized:
            self.initialize()

        # Извлекаем релевантные документы
        retrieval_results = self.hipporag.retrieve(
            queries=[query],
            num_to_retrieve=num_to_retrieve,
        )

        # Формируем контекст из найденных документов
        retrieved_docs = []
        if retrieval_results and len(retrieval_results) > 0:
            for result in retrieval_results[0].get("retrieved_docs", []):
                if isinstance(result, dict):
                    doc_text = result.get("text", result.get("content", str(result)))
                else:
                    doc_text = str(result)
                retrieved_docs.append(doc_text)

        # Создаём обогащённый промпт с контекстом
        context_text = "\n\n".join(retrieved_docs) if retrieved_docs else "Контекст не найден."

        enriched_query = f"""На основе следующего контекста ответь на вопрос:

Контекст:
{context_text}

Вопрос: {query}

Ответ:"""

        # Выполняем QA через HippoRAG
        qa_results = self.hipporag.rag_qa(queries=[enriched_query])

        if qa_results and len(qa_results) > 0:
            return qa_results[0].get("answer", qa_results[0].get("response", str(qa_results[0])))

        return "Не удалось получить ответ."


# Глобальный экземпляр сервиса
hipporag_service: Optional[HippoRAGService] = None


def get_hipporag_service() -> HippoRAGService:
    """Получение глобального экземпляра HippoRAG сервиса."""
    global hipporag_service
    if hipporag_service is None:
        hipporag_service = HippoRAGService()
    return hipporag_service


def init_hipporag_from_config(config: Dict[str, Any]) -> HippoRAGService:
    """
    Инициализация HippoRAG сервиса из конфигурации.

    Args:
        config: Словарь конфигурации HippoRAG.

    Returns:
        Инициализированный HippoRAG сервис.
    """
    global hipporag_service

    hipporag_config = config.get("hipporag", {})

    hipporag_service = HippoRAGService(
        save_dir=hipporag_config.get("save_dir", "outputs"),
        llm_model_name=hipporag_config.get("llm_model_name"),
        llm_base_url=hipporag_config.get("llm_base_url"),
        embedding_model_name=hipporag_config.get("embedding_model_name", "nvidia/NV-Embed-v2"),
        embedding_base_url=hipporag_config.get("embedding_base_url"),
    )

    return hipporag_service