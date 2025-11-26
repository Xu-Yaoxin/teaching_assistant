from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Optional, Generator
import json
import sqlite3
import requests
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import Query, Depends, HTTPException, FastAPI
from pydantic import BaseModel
import os
import re

try:
    from haystack.document_stores.in_memory import InMemoryDocumentStore
except ImportError:
    try:
        from haystack.document_stores import InMemoryDocumentStore
    except ImportError:
        class InMemoryDocumentStore:
            def __init__(self):
                self.storage = {}
                self.embeddings = {}
            
            def write_documents(self, documents):
                for doc in documents:
                    self.storage[doc.id] = doc
            
            def get_all_documents(self):
                return list(self.storage.values())
            
            def get_document_by_id(self, doc_id):
                return self.storage.get(doc_id)

from config import SIMILARITY_THRESHOLD, OLLAMA_BASE_URL, MODEL_NAME
from file_processor import load_documents
from knowledge_fetcher import fetch_wikipedia_knowledge
from topic_processor import topic_segment
from retrieval_utils import filter_docs_by_similarity
from pipeline import build_pipeline
from memory import ConversationMemory

app = FastAPI(title="QuestionAnsweringSystemAPI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_store: Optional[InMemoryDocumentStore] = None
pipeline_components: Optional[Dict] = None
single_memory: Optional[ConversationMemory] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_store, pipeline_components, single_memory

    document_store = InMemoryDocumentStore()

    pipeline_components = build_pipeline(document_store)

    single_memory = ConversationMemory()
    
    yield

app = FastAPI(
    title="QuestionAnsweringSystemAPI",
    version="1.0",
    lifespan=lifespan
)

def clean_think_tags(text):
    if not text:
        return text

    if len(text) < 50 and '<think>' not in text:
        return text

    cleaned = re.sub(r'/think/.*?/think/', '', text, flags=re.DOTALL)

    if not cleaned.strip():
        return text
    
    return cleaned


class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[Dict]] = None

class TitleGenerateRequest(BaseModel):
    content: str
    max_length: Optional[int] = 6
    min_length: Optional[int] = 1

class TitleGenerateResponse(BaseModel):
    success: bool
    title: str
    message: str

class DocumentLoadRequest(BaseModel):

    path: str

class StreamData:
    
    @staticmethod
    def answer_chunk(chunk: str) -> str:
        return json.dumps({
            "type": "answer",
            "content": chunk
        }, ensure_ascii=False)
    
    @staticmethod
    def references(references: List[Dict]) -> str:
        return json.dumps({
            "type": "references",
            "content": references
        }, ensure_ascii=False)
    
    @staticmethod
    def error(error: str) -> str:
        return json.dumps({
            "type": "error",
            "content": error
        }, ensure_ascii=False)
    
    @staticmethod
    def done(conversation_id: str) -> str:
        return json.dumps({
            "conversation_id": conversation_id,
            "done": True
        }, ensure_ascii=False)

def get_memory() -> ConversationMemory:
    global single_memory
    if single_memory is None:
        single_memory = ConversationMemory()
    return single_memory

@app.api_route("/ask-stream", methods=["POST", "OPTIONS"])
def ask_question_stream(
        request: QuestionRequest,
        memory: ConversationMemory = Depends(get_memory)
) -> StreamingResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="The question cannot be left blank.")

    global document_store, pipeline_components

    if not pipeline_components:
        pipeline_components = build_pipeline(document_store)

    def generate() -> Generator[str, None, None]:
        try:
            # 验证管道组件
            if not pipeline_components:
                yield f"{StreamData.error('The pipeline components have not been initialized. Please load the document first.')}\n"
                return
            
            pipeline = pipeline_components["pipeline"]
            text_embedder = pipeline_components["text_embedder"]
            retriever = pipeline_components["retriever"]
            prompt_builder = pipeline_components["prompt_builder"]
            generator = pipeline_components["generator"]

            if request.history:
                chat_history = []
                for msg in request.history:
                    if msg['role'] == 'user':
                        chat_history.append({'user': msg['content'], 'assistant': ''})
                    elif msg['role'] == 'assistant' and chat_history:
                        chat_history[-1]['assistant'] = msg['content']
            else:
                chat_history = memory.load()
            #print(f"历史对话{chat_history}")
            query, selected_messages, keyword = topic_segment(request.question, chat_history)
            print("---------------Historical dialogue filtering completed---------------")
            formatted_history = [
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in selected_messages
            ]
            print(f"ratio: {len(formatted_history)}/{len(chat_history)}")
            #print(f"content: {selected_messages}")
            print("---------------------------------------------")
            embedded_query = text_embedder.run(text=query)["embedding"]
            retrieved_docs = retriever.run(query_embedding=embedded_query)["documents"]
            wiki_docs = fetch_wikipedia_knowledge(keyword)

            all_docs_in_store = list(document_store.storage.values())
            doc_embeddings = {
                doc.id: doc.embedding
                for doc in all_docs_in_store
                if doc.embedding is not None
            }

            filtered_local_docs = filter_docs_by_similarity(
                docs=retrieved_docs,
                query_embedding=embedded_query,
                embedder=text_embedder,
                doc_embeddings=doc_embeddings,
                threshold=SIMILARITY_THRESHOLD
            )
            filtered_wiki_docs = filter_docs_by_similarity(
                docs=wiki_docs,
                query_embedding=embedded_query,
                embedder=text_embedder,
                doc_embeddings=None,
                threshold=SIMILARITY_THRESHOLD
            )
            all_docs = filtered_local_docs + filtered_wiki_docs
            prompt = prompt_builder.run(
                documents=all_docs,
                query=query,
                selected_history=formatted_history
            )["prompt"]
            print("---------------------------------------------")
            print(f"CurrentPrompt {prompt}")
            print("---------------------------------------------")
            full_answer = ""
            for chunk in generator.generate_stream(prompt):
                # if '<think>' in chunk or '</think>' in chunk:
                #     cleaned_chunk = clean_think_tags(chunk)
                #     if cleaned_chunk:
                #         full_answer += cleaned_chunk
                #         yield f"{StreamData.answer_chunk(cleaned_chunk)}\n"
                # else:
                #     full_answer += chunk
                #     yield f"{StreamData.answer_chunk(chunk)}\n"
                full_answer += chunk
                yield f"{StreamData.answer_chunk(chunk)}\n"

            references = []
            for doc in all_docs:
                doc_meta = doc.meta
                source_type = doc_meta.get("file_type") or doc_meta.get("type", "UNKNOWN")
                source_name = doc_meta.get("source", "unknown source")

                if source_type == "VIDEO" and "start_time" in doc_meta:
                    details = f"Fragment{doc_meta.get('chunk_id')} @ {doc_meta['start_time']:.1f}s"
                elif source_type == "PDF" and "page" in doc_meta:
                    details = f"Number{doc_meta['page']}page，Fragment{doc_meta.get('chunk_id')}"
                elif source_type == "PPT" and "slide" in doc_meta:
                    details = f"PPT{doc_meta['slide']}，Fragment{doc_meta.get('chunk_id')}"
                elif source_type == "WIKIPEDIA":
                    details = f"Item：{doc_meta.get('title')}"
                else:
                    details = f"Fragment{doc_meta.get('chunk_id')}"
                
                references.append({
                    "type": source_type,
                    "source": source_name,
                    "details": details
                })
            
            yield f"{StreamData.references(references)}\n"

            # if not request.history:
            #     memory.add_user_message(request.question)
            #     memory.add_assistant_message(full_answer)
            memory.add_user_message(query)
            memory.add_assistant_message(clean_think_tags(full_answer))

            yield f"{StreamData.done('current_conversation_id')}\n"
            
        except Exception as e:
            error_msg = f"Request processing failedRequest processing failed：{str(e)}"
            print(f"API error: {error_msg}")
            yield f"{StreamData.error(error_msg)}\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )


@app.post(
    "/generate_title",
    response_model=TitleGenerateResponse,
    summary="Generate a text title",
    description="Receive the text content, call the large model to generate a core title that meets the length requirements, and support fallback processing."
)
def generate_title(
    request: TitleGenerateRequest,
):

    if not request.content.strip():
        raise HTTPException(status_code=400, detail="The text content for generating the title cannot be empty.")
    if request.min_length > request.max_length:
        raise HTTPException(status_code=400, detail="The minimum length cannot be greater than the maximum length.")
    
    # 调用大模型生成标题
    try:
        response = requests.post(
            url=f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": f"Generate a concise title for the following text（Not exceed{request.max_length}）：{request.content}",
                "stream": False,
                "temperature": 0.3,
                "top_p": 0.9
            },
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        generated_title = result.get("response", "").strip()
        
        if generated_title:
            return {
                "success": True,
                "title": generated_title,
                "message": str(len(generated_title))
            }
    except Exception as e:
        print(f"Title generation failedTitle generation failed: {str(e)}")
    
    # 降级方案
    fallback_title = request.content[:request.max_length].strip()
    if len(request.content) > request.max_length:
        fallback_title += "..."
    return {
        "success": False,
        "title": fallback_title,
        "message": "Title generation failedTitle generation failed"
    }

@app.post("/load-documents", response_model=Dict[str, str])
def load_documents_api(
        request: DocumentLoadRequest,
) -> Dict[str, str]:

    global document_store, pipeline_components
    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not exist: {request.path}")
    
    try:
        docs = load_documents(str(path))
        if not docs:
            return {
                "status": "warning",
                "message": f"No valid document has been loaded.: {request.path}"
            }
        if not pipeline_components:
            pipeline_components = build_pipeline(document_store)

        if "doc_embedder" in pipeline_components:
            doc_embedder = pipeline_components["doc_embedder"]
            embedded_docs = doc_embedder.run(documents=docs)["documents"]
            document_store.write_documents(embedded_docs)
        else:
            document_store.write_documents(docs)
        
        return {
            "status": "success",
            "message": f"Successfully loaded and processed {len(docs)} document Fragment, from: {request.path}"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process the document: {str(e)}"
        )

# 入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)