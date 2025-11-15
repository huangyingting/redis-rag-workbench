import json
import os
import os.path
from string import Template
from typing import Any, List, Optional

import gradio as gr
import vertexai
from datasets import Dataset
from google.auth import load_credentials_from_dict
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langchain_redis import RedisChatMessageHistory, RedisVectorStore
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness
from redisvl.extensions.llmcache import SemanticCache
from redisvl.extensions.router import SemanticRouter
from redisvl.utils.rerank import CohereReranker, HFCrossEncoderReranker
import yaml
from redisvl.utils.vectorize import (
    AzureOpenAITextVectorizer,
    HFTextVectorizer,
    OpenAITextVectorizer,
    vectorizer_from_dict,
)
from redisvl.utils.utils import create_ulid

from workbench.shared.cached_llm import CachedLLM
from workbench.shared.converters import str_to_bool
from workbench.shared.llm_utils import (
    LLMs,
    default_openai_embedding_model,
    default_openai_model,
    default_vertex_embedding_model,
    default_vertex_model,
    openai_embedding_models,
    openai_models,
    vertex_embedding_models,
    vertex_models,
)
from workbench.shared.logger import logger
from workbench.shared.pdf_manager import PDFManager, PDFMetadata
from workbench.shared.pdf_utils import process_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ChatApp:
    def __init__(self) -> None:
        self.session_id = None
        self.pdf_manager = None
        self.current_pdf_index = None  # Initialize the missing attribute

        self.redis_url = os.environ.get("REDIS_URL")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.cohere_api_key = os.environ.get("COHERE_API_KEY")
        self.azure_cohere_api_key = os.environ.get("AZURE_COHERE_API_KEY")
        self.azure_cohere_endpoint = os.environ.get("AZURE_COHERE_ENDPOINT")

        self.azure_openai_api_version = os.environ.get(
            "AZURE_OPENAI_API_VERSION"
        )  # ex: 2024-08-01-preview
        self.azure_openai_api_key = os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )  # ex: 1234567890abcdef
        self.azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.azure_openai_embedding_deployment = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        )

        self.gcloud_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.gcloud_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID")

        required_vars = {"REDIS_URL": self.redis_url}

        missing_vars = {k: v for k, v in required_vars.items() if not v}

        if missing_vars or not self._cohere_credentials_available():
            self.credentials_set = False
        else:
            self.credentials_set = True

        self.initialized = False
        self.RERANKERS = {}
        self.reranker_type = None

        # Initialize non-API dependent variables
        self.chunk_size = int(os.environ.get("DEFAULT_CHUNK_SIZE", 500))
        self.chunking_technique = os.environ.get(
            "DEFAULT_CHUNKING_TECHNIQUE", "Recursive Character"
        )
        self.chat_history = None
        self.N = 0
        self.count = 0
        self.use_semantic_cache = str_to_bool(
            os.environ.get("DEFAULT_USE_SEMANTIC_CACHE")
        )
        self.use_rerankers = str_to_bool(os.environ.get("DEFAULT_USE_RERANKERS"))
        self.top_k = int(os.environ.get("DEFAULT_TOP_K", 3))
        self.distance_threshold = float(
            os.environ.get("DEFAULT_DISTANCE_THRESHOLD", 0.30)
        )
        self.llm_temperature = float(os.environ.get("DEFAULT_LLM_TEMPERATURE", 0.7))
        self.use_chat_history = str_to_bool(os.environ.get("DEFAULT_USE_CHAT_HISTORY"))
        self.use_semantic_router = str_to_bool(
            os.environ.get("DEFAULT_USE_SEMANTIC_ROUTER")
        )
        self.use_ragas = str_to_bool(os.environ.get("DEFAULT_USE_RAGAS"))

        self.available_llms = {}

        if self.openai_api_key is not None:
            self.available_llms[LLMs.openai] = openai_models()

        if self.azure_openai_deployment is not None:
            self.available_llms[LLMs.azure] = [self.azure_openai_deployment]

        if self.gcloud_credentials is not None:
            self.vertexai_credentials, _ = load_credentials_from_dict(
                json.loads(self.gcloud_credentials)
            )
            vertexai.init(
                project=self.gcloud_project_id, credentials=self.vertexai_credentials
            )
            self.available_llms[LLMs.vertexai] = vertex_models()

        self.llm_model_providers = list(self.available_llms.keys())

        self.available_embedding_models = {}

        if self.openai_api_key is not None:
            self.available_embedding_models[LLMs.openai] = openai_embedding_models()

        if self.azure_openai_embedding_deployment is not None:
            self.available_embedding_models[LLMs.azure] = [
                self.azure_openai_embedding_deployment
            ]

        if self.gcloud_credentials is not None:
            self.available_embedding_models[LLMs.vertexai] = vertex_embedding_models()

        self.embedding_model_providers = list(self.available_embedding_models.keys())
        if LLMs.openai in self.llm_model_providers:
            self.selected_llm_provider = LLMs.openai
            self.selected_llm = default_openai_model()
        elif LLMs.azure in self.llm_model_providers:
            self.selected_llm_provider = LLMs.azure
            self.selected_llm = self.azure_openai_deployment
        elif LLMs.vertexai in self.llm_model_providers:
            self.selected_llm_provider = LLMs.vertexai
            self.selected_llm = default_vertex_model()
        else:
            raise Exception(
                "You need to specify credentials for either OpenAI, Azure, or Google Cloud"
            )

        if LLMs.openai in self.embedding_model_providers:
            self.selected_embedding_model_provider = LLMs.openai
            self.selected_embedding_model = default_openai_embedding_model()
        elif LLMs.azure in self.embedding_model_providers:
            self.selected_embedding_model_provider = LLMs.azure
            self.selected_embedding_model = self.azure_openai_embedding_deployment
        elif LLMs.vertexai in self.embedding_model_providers:
            self.selected_embedding_model_provider = LLMs.vertexai
            self.selected_embedding_model = default_vertex_embedding_model()
        else:
            raise Exception(
                "You need to specify embedding credentials for OpenAI, Azure, or Google Cloud"
            )

        self.llm = None
        self.evalutor_llm = None
        self.cached_llm = None
        self.vector_store = None
        self.llmcache = None
        self.index_name = None
        self.semantic_router = None

    def _has_azure_cohere(self) -> bool:
        return bool(self.azure_cohere_api_key and self.azure_cohere_endpoint)

    def _cohere_credentials_available(self) -> bool:
        return bool(self.cohere_api_key) or self._has_azure_cohere()

    def _normalized_azure_cohere_endpoint(self) -> Optional[str]:
        if not self.azure_cohere_endpoint:
            return None
        return self.azure_cohere_endpoint.rstrip("/")

    def _cohere_reranker_params(self) -> tuple[Optional[str], dict]:
        if self._has_azure_cohere():
            base_url = self._normalized_azure_cohere_endpoint()
            client_kwargs = {}
            if base_url:
                client_kwargs["base_url"] = base_url
            return self.azure_cohere_api_key, client_kwargs
        if self.cohere_api_key:
            return self.cohere_api_key, {}
        return None, {}

    def _load_router_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            template = Template(file.read())
        try:
            resolved = template.substitute(os.environ)
        except KeyError as exc:  # pragma: no cover - configuration guard
            raise ValueError(
                f"Missing environment variable {exc} required by router.yaml"
            ) from exc

        data = yaml.safe_load(resolved)
        if not isinstance(data, dict):
            raise ValueError("Router configuration must be a mapping")
        return data

    def _build_router_vectorizer(self, vectorizer_config: dict):
        if not vectorizer_config:
            return None

        vectorizer_type = vectorizer_config.get("type")
        api_config = vectorizer_config.get("api_config")

        if vectorizer_type == "azure_openai":
            effective_config = api_config or {}
            if not effective_config:
                effective_config = {
                    "api_key": self.azure_openai_api_key,
                    "api_version": self.azure_openai_api_version,
                    "azure_endpoint": self.azure_openai_endpoint,
                }
            return AzureOpenAITextVectorizer(
                model=vectorizer_config.get("model"),
                api_config=effective_config,
            )

        return vectorizer_from_dict(vectorizer_config)

    def _semantic_cache_vectorizer(self):
        provider = self.selected_embedding_model_provider
        try:
            if provider == LLMs.azure:
                if not all(
                    [
                        self.azure_openai_api_key,
                        self.azure_openai_api_version,
                        self.azure_openai_endpoint,
                    ]
                ):
                    raise ValueError(
                        "Azure OpenAI credentials are required for semantic cache vectorizer"
                    )
                return AzureOpenAITextVectorizer(
                    model=self.selected_embedding_model,
                    api_config={
                        "api_key": self.azure_openai_api_key,
                        "api_version": self.azure_openai_api_version,
                        "azure_endpoint": self.azure_openai_endpoint,
                    },
                )

            if provider == LLMs.openai:
                if not self.openai_api_key:
                    raise ValueError("OPENAI_API_KEY must be set for semantic cache")
                return OpenAITextVectorizer(
                    model=self.selected_embedding_model,
                    api_config={"api_key": self.openai_api_key},
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Failed to configure semantic cache vectorizer for %s: %s. Falling back to default.",
                provider,
                exc,
            )

        return HFTextVectorizer(model="redis/langcache-embed-v1")

    def _initialize_semantic_router(self):
        try:
            logger.info("Initializing semantic router")
            router_config = self._load_router_config("workbench/router.yaml")
            vectorizer_config = router_config.pop("vectorizer", {})
            vectorizer = self._build_router_vectorizer(vectorizer_config)

            self.semantic_router = SemanticRouter(
                vectorizer=vectorizer,
                redis_url=self.redis_url,
                overwrite=True,
                **router_config,
            )
            logger.info("Semantic router initialized")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to initialize semantic router: %s", exc)
            self.semantic_router = None
            raise

    def initialize(self):
        if self.credentials_set:
            self.initialize_components()

    def initialize_components(self):
        if not self.credentials_set:
            raise ValueError("Credentials must be set before initializing components")

        self.pdf_manager = PDFManager(self.redis_url)

        # Perform data reconciliation to ensure consistency
        logger.info("Performing data reconciliation...")
        try:
            fixed, removed, orphaned = self.pdf_manager.reconcile_data()
            if fixed > 0 or removed > 0 or orphaned > 0:
                logger.info(f"Reconciliation summary - Fixed: {fixed}, Removed: {removed}, Orphaned cleaned: {orphaned}")
            else:
                logger.info("Data reconciliation complete - no issues found")
        except Exception as e:
            logger.warning(f"Data reconciliation failed: {e}")

        # Initialize rerankers
        logger.info("Initializing rerankers")
        self.RERANKERS = {}

        cohere_api_key, cohere_client_kwargs = self._cohere_reranker_params()
        if cohere_api_key:
            logger.info(
                "Enabling Cohere reranker via %s credentials",
                "Azure" if self._has_azure_cohere() else "Cohere",
            )
            self.RERANKERS["Cohere"] = CohereReranker(
                limit=3,
                api_config={"api_key": cohere_api_key},
                **cohere_client_kwargs,
            )
        else:
            logger.info("Cohere credentials missing. Falling back to HuggingFace reranker.")
            self.RERANKERS["HuggingFace"] = HFCrossEncoderReranker(
                "BAAI/bge-reranker-base"
            )

        logger.info("Rerankers initialized")
        self._ensure_reranker_type()

        # Init semantic router only if enabled
        if self.use_semantic_router:
            self._initialize_semantic_router()
        else:
            self.semantic_router = None
            logger.info("Semantic router disabled")

        # Init chat history if use_chat_history is True
        if self.use_chat_history:
            self.chat_history = RedisChatMessageHistory(
                session_id=self.session_id, redis_url=self.redis_url
            )
        else:
            self.chat_history = None

        # Init LLM
        self.update_llm()

        self.initialized = True

    def initialize_session(self):
        self.session_id = create_ulid()
        if self.use_chat_history:
            self.chat_history = RedisChatMessageHistory(
                session_id=self.session_id,
                redis_url=self.redis_url,
                index_name="chat_history",  # Use a common index for all chat histories
            )
        else:
            self.chat_history = None

        return {"session_id": self.session_id, "chat_history": self.chat_history}

    def set_credentials(
        self,
        redis_url,
        openai_key,
        cohere_key,
        azure_cohere_key,
        azure_cohere_endpoint,
    ):
        self.redis_url = redis_url or self.redis_url
        self.openai_api_key = openai_key or self.openai_api_key
        self.cohere_api_key = cohere_key or self.cohere_api_key
        self.azure_cohere_api_key = azure_cohere_key or self.azure_cohere_api_key
        self.azure_cohere_endpoint = (
            azure_cohere_endpoint or self.azure_cohere_endpoint
        )

        # Update environment variables
        if self.redis_url:
            os.environ["REDIS_URL"] = self.redis_url
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.cohere_api_key:
            os.environ["COHERE_API_KEY"] = self.cohere_api_key
        if self.azure_cohere_api_key:
            os.environ["AZURE_COHERE_API_KEY"] = self.azure_cohere_api_key
        if self.azure_cohere_endpoint:
            os.environ["AZURE_COHERE_ENDPOINT"] = self.azure_cohere_endpoint

        self.credentials_set = bool(self.redis_url) and self._cohere_credentials_available()

        if self.credentials_set:
            self.initialize_components()

        return "Credentials updated successfully. You can now use the demo."

    def get_llm(self):
        """Get the right LLM based on settings and config."""
        match self.selected_llm_provider:
            case LLMs.azure:
                try:
                    model = AzureChatOpenAI(
                        azure_deployment=self.selected_llm,
                        api_version=self.azure_openai_api_version,
                        api_key=self.azure_openai_api_key,
                        azure_endpoint=self.azure_openai_endpoint,
                        temperature=self.llm_temperature,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error initializing Azure OpenAI model: {e} - must provide credentials for deployment"
                    )
            case LLMs.vertexai:
                try:
                    model = ChatVertexAI(
                        model=self.selected_llm,
                        temperature=self.llm_temperature,
                        max_tokens=None,
                        stop=None,
                        max_retries=2,
                        credentials=self.vertexai_credentials,
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error initializing VertexAI model: {e} - must provide credentials for deployment"
                    )
            case _:
                model = ChatOpenAI(
                    model=self.selected_llm,
                    temperature=0,
                )

        return model

    def get_embedding_model(self):
        """Get the right embedding model based on settings and config"""
        print(
            f"Generating embeddings for provider: {self.selected_embedding_model_provider} and model: {self.selected_embedding_model}"
        )
        match self.selected_embedding_model_provider:
            case LLMs.azure:
                return AzureOpenAIEmbeddings(
                    model=self.selected_embedding_model,
                    api_key=self.azure_openai_api_key,
                    api_version=self.azure_openai_api_version,
                    azure_endpoint=self.azure_openai_endpoint,
                )
            case LLMs.vertexai:
                return VertexAIEmbeddings(
                    model=self.selected_embedding_model,
                    credentials=self.vertexai_credentials,
                )
            case _:
                return OpenAIEmbeddings(api_key=self.openai_api_key)

    def get_reranker_choices(self):
        if self.initialized:
            return list(self.RERANKERS.keys())
        # Default choices before initialization
        return ["HuggingFace", "Cohere"]

    def __call__(self, file: str, chunk_size: int, chunking_technique: str) -> Any:
        """Process a file upload directly - used by the UI."""
        self.chunk_size = chunk_size
        self.chunking_technique = chunking_technique

        # First store the PDF and get its index
        return self.process_pdf(
            file, chunk_size, chunking_technique, self.selected_embedding_model
        )

    def build_chain(self, history: List[gr.ChatMessage]):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

        messages = [
            (
                "system",
                """You are a helpful AI assistant. Use the following pieces of
                    context to answer the user's question. If you don't know the
                answer, just say that you don't know, don't try to make up an
                    answer. Please be as detailed as possible with your
                    answers.""",
            ),
            ("system", "Context: {context}"),
        ]

        if self.use_chat_history:
            for msg in history:
                messages.append((msg["role"], msg["content"]))

        messages.append(("human", "{input}"))
        messages.append(
            (
                "system",
                "Provide a helpful and accurate answer based on the given context and question:",
            )
        )
        prompt = ChatPromptTemplate.from_messages(messages)

        combine_docs_chain = create_stuff_documents_chain(self.cached_llm, prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        return rag_chain

    def update_chat_history(
        self, history: List[gr.ChatMessage], use_chat_history: bool, session_state
    ):
        self.use_chat_history = use_chat_history

        if session_state is None:
            session_state = self.initialize_session()

        if self.use_chat_history:
            if (
                "chat_history" not in session_state
                or session_state["chat_history"] is None
            ):
                session_state["chat_history"] = RedisChatMessageHistory(
                    session_id=session_state.get("session_id", create_ulid()),
                    redis_url=self.redis_url,
                    index_name="chat_history",
                )
        else:
            if "chat_history" in session_state and session_state["chat_history"]:
                try:
                    session_state["chat_history"].clear()
                except Exception as e:
                    print(f"DEBUG: Error clearing chat history: {str(e)}")
            session_state["chat_history"] = None

        history.clear()

        return history, session_state

    def get_chat_history(self):
        if self.chat_history and self.use_chat_history:
            messages = self.chat_history.messages
            formatted_history = []
            for msg in messages:
                if msg.type == "human":
                    formatted_history.append(f"👤 **Human**: {msg.content}\n")
                elif msg.type == "ai":
                    formatted_history.append(f"🤖 **AI**: {msg.content}\n")
            return "\n".join(formatted_history)
        return "No chat history available."

    def update_semantic_router(self, use_semantic_router: bool):
        self.use_semantic_router = use_semantic_router
        if self.use_semantic_router:
            if self.semantic_router is None:
                try:
                    self._initialize_semantic_router()
                except Exception:  # pylint: disable=broad-except
                    logger.warning(
                        "Disabling semantic router after initialization failure"
                    )
                    self.use_semantic_router = False
        else:
            self.semantic_router = None

    def update_ragas(self, use_ragas: bool):
        self.use_ragas = use_ragas

    def update_reranker_usage(self, use_rerankers: bool):
        self.use_rerankers = use_rerankers

    def update_reranker_type(self, reranker_type: str):
        if reranker_type in self.RERANKERS:
            self.reranker_type = reranker_type

    def _ensure_reranker_type(self):
        if not self.RERANKERS:
            self.reranker_type = None
            return

        if self.reranker_type not in self.RERANKERS:
            self.reranker_type = next(iter(self.RERANKERS))

    def update_llm(self):
        self.llm = self.get_llm()
        self.evalutor_llm = LangchainLLMWrapper(self.llm)

        if self.use_semantic_cache:
            self.cached_llm = CachedLLM(self.llm, self.llmcache)
        else:
            self.cached_llm = self.llm

    def update_model(self, new_model: str, new_model_provider: str):
        self.selected_llm = new_model
        self.selected_llm_provider = new_model_provider
        self.update_llm()

    def update_temperature(self, new_temperature: float):
        self.llm_temperature = new_temperature
        self.update_llm()

    def update_top_k(self, new_top_k: int):
        self.top_k = new_top_k

    def make_semantic_cache(self) -> SemanticCache:
        semantic_cache_index_name = f"llmcache:{self.index_name}"
        vectorizer = self._semantic_cache_vectorizer()
        return SemanticCache(
            name=semantic_cache_index_name,
            redis_url=self.redis_url,
            distance_threshold=self.distance_threshold,
            vectorizer=vectorizer,
        )

    def clear_semantic_cache(self):
        # Always make a new SemanticCache in case use_semantic_cache is False
        semantic_cache = self.make_semantic_cache()
        semantic_cache.clear()

    def update_semantic_cache(self, use_semantic_cache: bool):
        self.use_semantic_cache = use_semantic_cache
        if self.use_semantic_cache and self.index_name:
            self.llmcache = self.make_semantic_cache()
        else:
            self.llmcache = None

        self.update_llm()

    def update_distance_threshold(self, new_threshold: float):
        self.distance_threshold = new_threshold
        if self.index_name:
            self.llmcache = self.make_semantic_cache()
            self.update_llm()

    def get_last_cache_status(self) -> bool:
        if isinstance(self.cached_llm, CachedLLM):
            return self.cached_llm.get_last_cache_status()
        return False

    def rerank_results(self, query, results):
        if not self.use_rerankers or not self.reranker_type:
            return results, None, None

        reranker = self.RERANKERS[self.reranker_type]
        original_results = [r.page_content for r in results]

        reranked_results, scores = reranker.rank(query=query, docs=original_results)

        # Reconstruct the results with reranked order, using fuzzy matching
        reranked_docs = []
        for reranked in reranked_results:
            reranked_content = (
                reranked["content"] if isinstance(reranked, dict) else reranked
            )
            best_match = max(
                results, key=lambda r: self.similarity(r.page_content, reranked_content)
            )
            reranked_docs.append(best_match)

        rerank_info = {
            "original_order": original_results,
            "reranked_order": [
                r["content"] if isinstance(r, dict) else r for r in reranked_results
            ],
            "original_scores": [1.0]
            * len(results),  # Assuming original scores are not available
            "reranked_scores": scores,
        }

        return reranked_docs, rerank_info, original_results

    def similarity(self, s1, s2):
        # Simple similarity measure based on common words
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        return len(words1.intersection(words2)) / len(words1.union(words2))

    def rerankers(self):
        return self.RERANKERS

    def evaluate_response(self, query, result):
        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [result["answer"]],
                "contexts": [[c.page_content for c in result["context"]]],
            }
        )

        try:
            eval_results = evaluate(
                dataset=ds,
                metrics=[faithfulness, answer_relevancy],
                llm=self.evalutor_llm,
            )

            return eval_results
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            return {}

    def update_embedding_model_provider(self, new_provider: str):
        self.selected_embedding_model_provider = new_provider

    def process_pdf(
        self,
        file,
        chunk_size: int,
        chunking_technique: str,
        selected_embedding_model: str,
    ) -> Any:
        """Process a new PDF file upload."""
        try:
            print(f"Using selected_embedding_model: {selected_embedding_model}")
            embeddings = self.get_embedding_model()

            # Let PDFManager handle complete processing
            self.index_name = self.pdf_manager.process_pdf_complete(
                file, chunk_size, chunking_technique, embeddings
            )
            self.current_pdf_index = self.index_name  # Set current_pdf_index
            self.selected_embedding_model = selected_embedding_model

            # Load the vector store that was just created
            self.vector_store = self.pdf_manager.load_pdf_complete(self.index_name, embeddings)
            self.update_semantic_cache(self.use_semantic_cache)

        except Exception as e:
            logger.error(f"Error during process_pdf: {e}")
            raise

    def load_pdf(self, index_name: str) -> bool:
        """Load a previously processed PDF."""
        try:
            embeddings = self.get_embedding_model()

            # Let PDFManager handle complete loading (with reprocessing if needed)
            self.vector_store = self.pdf_manager.load_pdf_complete(index_name, embeddings)

            # Update app state
            self.index_name = index_name
            self.current_pdf_index = index_name  # Set current_pdf_index
            metadata = self.pdf_manager.get_pdf_metadata(index_name)
            if metadata:
                self.chunk_size = metadata.chunk_size
                self.chunking_technique = metadata.chunking_technique

            self.update_semantic_cache(self.use_semantic_cache)
            return True

        except Exception as e:
            logger.error(f"Failed to load PDF {index_name}: {e}")
            return False

    def search_pdfs(self, query: str = "") -> List[PDFMetadata]:
        """Search available PDFs."""
        return self.pdf_manager.search_pdfs(query)

    def get_pdf_file(self, index_name: str) -> Optional[str]:
        """Get the file path for a stored PDF."""
        return self.pdf_manager.get_pdf_file(index_name)


def generate_feedback(evaluation_scores):
    if not evaluation_scores:
        return "RAGAS evaluation failed."

    feedback = ["RAGAS Metrics:"]
    for metric, score in evaluation_scores.items():
        feedback.append(f"  - {metric}: {score:.4f}")
    return "\n".join(feedback)
