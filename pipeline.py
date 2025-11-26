from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.utils.auth import Secret
from config import OLLAMA_BASE_URL, MODEL_NAME, EMBEDDING_MODEL, RETRIEVE_TOP_K
from typing import Generator, Optional
import requests
import json


class OpenAIGenerator:
    def __init__(
            self,
            api_base_url: str,
            api_key: Secret,
            model: str,
            generation_kwargs: Optional[dict] = None
    ):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model = model
        self.generation_kwargs = generation_kwargs or {}

        if not self.api_base_url.endswith('/'):
            self.api_base_url += '/'

    def generate_stream(
            self,
            prompt: str
    ) -> Generator[str, None, None]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {str(self.api_key)}"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True, **self.generation_kwargs
        }

        try:
            with requests.post(
                    url=f"{self.api_base_url}chat/completions",
                    headers=headers,
                    json=data,
                    stream=True
            ) as response:
                response.raise_for_status()

                is_reasoning_active = False

                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8').lstrip('data: ')
                        if line == '[DONE]':
                            if is_reasoning_active:
                                yield "/think/"
                                is_reasoning_active = False
                            break

                        try:
                            chunk = json.loads(line)
                            reasoning = chunk.get('choices', [{}])[0].get('delta', {}).get('reasoning', '')
                            content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')

                            if reasoning:
                                if not is_reasoning_active:
                                    yield "/think/"
                                    is_reasoning_active = True
                                yield reasoning

                            elif content:
                                if is_reasoning_active:
                                    yield "/think/"
                                    is_reasoning_active = False
                                yield content

                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            yield f"An error occurred during the request: {str(e)}"


def build_pipeline(document_store: InMemoryDocumentStore):

    doc_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
    doc_embedder.warm_up()

    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
    text_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=RETRIEVE_TOP_K
    )

    generator = OpenAIGenerator(
        api_base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key=Secret.from_token("ollama"),  # 保持Haystack的Secret类型
        model=MODEL_NAME,
        generation_kwargs={
            "temperature": 0.7,
            "max_tokens": 10240
        }
    )

    # 2. 定义Prompt模板
    prompt_template = """
    You are an intelligent question-answering assistant. Please refer to historical conversation records and the following documentation to answer user questions concisely and accurately.

    Historical Dialogue：
    {% for msg in selected_history %}
    - {{ msg }}
    {% endfor %}

    Document：
    {% for doc in documents %}
    - {{ doc.content }} (Source: {{ doc.meta.source }}{% if doc.meta.start_time %}, time: {{ doc.meta.start_time|round(1) }}s{% endif %})
    {% endfor %}

    Current question：{{ query }}
    Answer：
    """
    prompt_builder = PromptBuilder(
        template=prompt_template,
        required_variables=["documents", "query", "selected_history"]
    )

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    return {
        "pipeline": pipeline,
        "doc_embedder": doc_embedder,
        "text_embedder": text_embedder,
        "retriever": retriever,
        "prompt_builder": prompt_builder,
        "generator": generator
    }
