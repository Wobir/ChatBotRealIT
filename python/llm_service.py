"""Модуль для работы с LLM-сервисом: инициализация, генерация ответов и мониторинг конфигов."""
import uuid
import traceback
import time
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from pathlib import Path

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

import python.loader as loader
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

llm: Optional[AsyncLLM] = None
system_prompt: str = ""


def load_LLM():
    """Загружает модель LLM на основе конфигурации, если она ещё не загружена или изменилась."""
    global llm
    if loader.MODEL_PATH is None:
        raise RuntimeError("MODEL_PATH не установлен. Проверь config.yaml -> paths.model")

    cfg = loader.load_yaml(loader.MODEL_PATH)

    engine_args = AsyncEngineArgs(
        model=cfg["model"].get("name", "PrunaAI/IlyaGusev-saiga_mistral_7b_merged-AWQ-4bit-smashed"),
        quantization=cfg["model"].get("quantization", "awq_marlin"),
        gpu_memory_utilization=cfg["model"].get("gpu_memory_utilization", 0.6),
        dtype=cfg["model"].get("dtype", "auto"),
        max_model_len=cfg["model"].get("max_model_len", None),
    )

    llm = AsyncLLM.from_engine_args(engine_args)
    print(f"Загружена модель: {engine_args.model}")


def init_hipporag():
    """Инициализация HippoRAG сервиса на основе конфигурации."""
    global hipporag_service
    try:
        from python.hipporag_service import init_hipporag_from_config, get_hipporag_service

        config = loader.config
        if "hipporag" in config:
            hipporag_service = init_hipporag_from_config(config)
            print("HippoRAG сервис инициализирован")

            # Автоматическая индексация документов если включена
            if config["hipporag"].get("auto_index", False):
                index_documents_from_configs()
        else:
            print("HippoRAG не настроен в config.yaml")
    except Exception as e:
        print(f"Ошибка инициализации HippoRAG: {e}")
        traceback.print_exc()


def index_documents_from_configs():
    """Индексация документов из конфигурационных файлов в HippoRAG."""
    global hipporag_service

    if not hipporag_service or not hipporag_service._initialized:
        return

    try:
        docs = []

        # Собираем документы из всех YAML конфигов
        for config_file in ["configs/courses.yaml", "configs/locations.yaml",
                           "configs/teachers.yaml", "configs/examples.yaml"]:
            config_path = Path(config_file)
            if config_path.exists():
                cfg = loader.load_yaml(str(config_path))

                if isinstance(cfg, dict):
                    for key, value in cfg.items():
                        if isinstance(value, str):
                            docs.append(f"{key}: {value}")
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                docs.append(f"{key} - {sub_key}: {sub_value}")
                        elif isinstance(value, list):
                            for item in value:
                                docs.append(f"{key}: {item}")
                elif isinstance(cfg, list):
                    for item in cfg:
                        if isinstance(item, dict):
                            doc_parts = []
                            for k, v in item.items():
                                doc_parts.append(f"{k}: {v}")
                            docs.append(" | ".join(doc_parts))
                        else:
                            docs.append(str(item))

        # Добавляем system prompt как документ
        system_prompt_path = Path("configs/system_prompt.txt")
        if system_prompt_path.exists():
            docs.append(f"System Prompt: {system_prompt_path.read_text(encoding='utf-8')}")

        if docs:
            print(f"Индексация {len(docs)} документов в HippoRAG...")
            hipporag_service.index_documents(docs)
            print("Индексация завершена")
        else:
            print("Нет документов для индексации")

    except Exception as e:
        print(f"Ошибка при индексации документов: {e}")
        traceback.print_exc()


def build_system_prompt() -> str:
    """Собирает все блоки данных в один system_prompt."""
    global system_prompt
    data_blocks = loader.load_data_blocks()
    system_prompt = "\n\n".join(data_blocks.values())
    print(f"System prompt сформирован, длина: {len(system_prompt)} символов")
    return system_prompt


class ConfigWatcher(FileSystemEventHandler):
    """Автоматическая перезагрузка конфигурации при изменении файлов с дебаунсом."""
    last_event_time = 0

    def on_modified(self, event):
        if time.time() - self.last_event_time < 1:
            return
        self.last_event_time = time.time()
        try:
            path = Path(event.src_path)
            if path.name == loader.CONFIG_PATH.name:
                loader.load_paths()
                loader.load_sampling_params()
                load_LLM()
                build_system_prompt()
            elif loader.SAMPLES_PATH and path.name == loader.SAMPLES_PATH.name:
                loader.load_sampling_params()
            elif loader.MODEL_PATH and path.name == loader.MODEL_PATH.name:
                load_LLM()
            elif path in loader.DATA_PATHS.values():
                build_system_prompt()
        except Exception:
            print("Ошибка в ConfigWatcher:")
            traceback.print_exc()


def generate_answer(prompt: str, sampling_params: SamplingParams, request_id: Optional[str] = None):
    """Синхронная оболочка для генерации ответа."""
    if llm is None:
        raise RuntimeError("LLM не загружена. Проверь конфиг.")
    if request_id is None:
        request_id = f"chat-{uuid.uuid4().hex}"
    return llm.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params
    )


async def warmup_llm():
    """Прогрев LLM для ускорения первого запроса."""
    if llm is None:
        raise RuntimeError("LLM не загружена.")
    try:
        async for _ in generate_answer("Привет", SamplingParams(max_tokens=1)):
            break
        print("LLM прогрета")
    except Exception as e:
        print(f"Ошибка при прогреве LLM: {e}")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app):
    """Инициализация LLM при запуске приложения и наблюдение за конфигами."""
    try:
        loader.load_paths()
        loader.load_sampling_params()
        build_system_prompt()
        load_LLM()
        await warmup_llm()
        print("LLM инициализирован")
    except Exception:
        print("Ошибка при инициализации LLM:")
        traceback.print_exc()

    event_handler = ConfigWatcher()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=True)
    observer.start()

    yield

    observer.stop()
    observer.join()


async def get_llm_reply(context: list, request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Генерация ответа LLM на основе контекста с использованием HippoRAG для RAG."""
    from python.hipporag_service import hipporag_service as rag_service

    # Извлекаем последний вопрос пользователя
    last_user_message = ""
    for msg in reversed(context[-10:]):
        role = getattr(msg, "role", None) or getattr(msg, "type", None)
        if role == "user":
            last_user_message = getattr(msg, "content", "")
            break

    # Если HippoRAG инициализирован, используем его для поиска релевантных документов
    retrieved_context = ""
    if rag_service and rag_service._initialized and last_user_message:
        try:
            retrieval_results = rag_service.hipporag.retrieve(
                queries=[last_user_message],
                num_to_retrieve=3,
            )

            if retrieval_results and len(retrieval_results) > 0:
                retrieved_docs = []
                for result in retrieval_results[0].get("retrieved_docs", []):
                    if isinstance(result, dict):
                        doc_text = result.get("text", result.get("content", str(result)))
                    else:
                        doc_text = str(result)
                    retrieved_docs.append(doc_text)

                if retrieved_docs:
                    retrieved_context = "\n\n".join(retrieved_docs)
        except Exception as e:
            print(f"Ошибка при использовании HippoRAG: {e}")

    # Формируем промпт
    prompt_parts = [system_prompt]

    # Добавляем retrieved контекст если есть
    if retrieved_context:
        prompt_parts.append(f"\n\nРелевантная информация из базы знаний:\n{retrieved_context}")

    prompt_parts.append("\n\nДиалог с пользователем:")
    for msg in context[-10:]:
        role = getattr(msg, "role", None) or getattr(msg, "type", None)
        content = getattr(msg, "content", None)
        if role == "user":
            prompt_parts.append(f"Пользователь: {content}")
        elif role == "bot":
            prompt_parts.append(f"Бот: {content}")
    prompt_parts.append("Бот:")

    prompt = "\n".join(prompt_parts)

    async for output in generate_answer(prompt, loader.sampling_params, request_id):
        for completion in output.outputs:
            if completion.text:
                yield completion.text