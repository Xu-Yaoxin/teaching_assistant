from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Dict, Optional, Generator, Tuple
import json
from haystack.document_stores.in_memory import InMemoryDocumentStore
from config import SIMILARITY_THRESHOLD
from file_processor import load_documents
from knowledge_fetcher import fetch_wikipedia_knowledge
from topic_processor import topic_segment
from retrieval_utils import filter_docs_by_similarity
from pipeline import build_pipeline
from personalization import get_question_attention_level
from contextlib import asynccontextmanager
import sqlite3
from fastapi import Query, Depends, HTTPException, FastAPI, UploadFile, File, Request
from memory import ConversationMemory
from pydantic import BaseModel
import requests
from config import OLLAMA_BASE_URL, MODEL_NAME
import io
from pathlib import Path
from PIL import Image
import pytesseract
import PyPDF2
from unstructured.partition.auto import partition


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

class QuestionRequest(BaseModel):
    question: str
    difficulty: int = 2

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

class OCRResponse(BaseModel):
    success: bool
    result: str
    message: str

class TitleGenerateRequest(BaseModel):
    content: str
    max_length: Optional[int] = 6
    min_length: Optional[int] = 1

class TitleGenerateResponse(BaseModel):
    success: bool
    title: str
    message: str

class SimilarMathQuestionRequest(BaseModel):
    question: str

class SimilarMathQuestionResponse(BaseModel):
    success: bool
    similar_questions: List[str]
    total_count: int
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_store, pipeline_components, single_memory
    document_store = InMemoryDocumentStore()

    single_memory = ConversationMemory()
    yield

app = FastAPI(
    title="QuestionAnsweringSystemAPI",
    version="1.0",
    lifespan=lifespan
)

def get_db_connection():
    conn = sqlite3.connect("chat_app.db")
    conn.row_factory = sqlite3.Row
    return conn

def get_memory() -> ConversationMemory:
    global single_memory
    if single_memory is None:
        single_memory = ConversationMemory()
    return single_memory

@app.post("/load-conversation", response_model=Dict[str, List[Dict]])
def load_conversation(
    table_name: str = Query(..., description="Conversation content table name, in the format of conversation_YYYYMMDD_HHMMSS"),
    memory: ConversationMemory = Depends(get_memory)
):

    conn = get_db_connection()
    try:
        meta_cursor = conn.execute(
            "SELECT id FROM conversation_metadata WHERE table_name = ?",
            (table_name,)
        )
        meta_exists = meta_cursor.fetchone() is not None
        table_exists = False
        if meta_exists:
            table_cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            table_exists = table_cursor.fetchone() is not None
        if not (meta_exists and table_exists):
            memory.clear()
            return {
                "message": f"The dialogue table {table_name} does not exist. It has been reset to an empty dialogue.",
                "conversations": [],
                "loaded_count": 0
            }

        try:
            msg_cursor = conn.execute(f"SELECT id, sentence FROM {table_name} ORDER BY id")
            messages = msg_cursor.fetchall()
        except sqlite3.OperationalError:
            memory.clear()
            return {
                "message": f"The structure of the {table_name} dialogue table is incorrect. It has been reset to a blank dialogue.",
                "conversations": [],
                "loaded_count": 0
            }

        memory.clear()
        user_msg = None

        for msg in messages:
            msg_id = msg["id"]
            content = msg["sentence"]

            if msg_id == 1:
                continue

            if msg_id % 2 == 0:
                user_msg = content
            else:
                if user_msg is not None:
                    memory.add_user_message(user_msg)
                    memory.add_assistant_message(content)
                    user_msg = None

        if user_msg is not None:
            memory.add_user_message(user_msg)

        loaded_conversations = memory.load()
        return {
            "message": f"Successfully loaded the conversation. {table_name}",
            "conversations": loaded_conversations,
            "loaded_count": len(loaded_conversations)
        }

    finally:
        conn.close()

@app.post("/ask-stream")
def ask_question_stream(
        request: QuestionRequest,
        memory: ConversationMemory = Depends(get_memory)
) -> StreamingResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    difficulty_map = {
        1: "For beginners, explain in the simplest and most straightforward language without using any technical terms. Break down the steps clearly and use relatable examples to aid understanding.",
        2: "For ordinary scholars, answer in concise and accurate language, with a clear logic. Use appropriate basic terms and ensure the content is easy to understand and complete.",
        3: "For advanced scholars, use professional and rigorous academic language. Be able to flexibly employ deep professional terms to deeply analyze the essence of the problem and provide rigorous derivations and expansion ideas."
    }

    attention_map = {
        2: "The student has a basic understanding of the knowledge points but has minor doubts or unclear problem-solving思路. You should answer according to the following process: "
           "1. Analyze the problem by breaking it down into 'Understanding the Problem → Connecting to Knowledge Points → Problem-Solving Steps'. At each step, clearly identify the core of the corresponding knowledge point. For example: 'This question examines the XX knowledge point. The given conditions A and B in the question align with the application scenario of this knowledge point; therefore, the first step is to determine the core formula.' "
           "2. Clearly present each step of the solution process, marking key logical nodes (e.g., 'Note the constraints of condition XX here; otherwise, it will lead to deviations in the steps'). "
           "3. Provide targeted reminders of common pitfalls in similar problems (e.g., 'unit consistency', 'signs in formulas'), without expanding on unrelated content;",

        3: "The student has a weak understanding of the knowledge points, exhibits conceptual confusion, unclear core logic, or has repeatedly made mistakes on similar problems. You should answer according to the following process: "
           "1. First, restate the core concept of the knowledge point in plain language, supplemented by real-life examples to aid understanding. For example: 'The core of the XX theorem is that 'if condition A is met, result B can be derived', similar to the logic of 'you must finish your homework before you can engage in recreational activities'.' "
           "2. Break down the problem into 'minimum actionable steps'. For each step, detailedly explain 'why this is done' and 'the basis in knowledge points'. For example: 'This step uses the XX formula because the given conditions in the question satisfy the three applicable premises of this formula.' "
           "3. Summarize a general problem-solving template for similar questions (e.g., 'Identify key conditions through understanding the problem → Match with knowledge points/formulas → Calculate step by step → Verify the result') to solidify the problem-solving logic;"
    }

    difficulty_desc = difficulty_map.get(request.difficulty, difficulty_map[2])

    _,attention_level = get_question_attention_level(request.question)
    attention = attention_map.get(attention_level, "")

    def generate() -> Generator[str, None, None]:
        try:
            pipeline = pipeline_components["pipeline"]
            text_embedder = pipeline_components["text_embedder"]
            retriever = pipeline_components["retriever"]
            prompt_builder = pipeline_components["prompt_builder"]
            generator = pipeline_components["generator"]

            chat_history = memory.load()
            query, selected_messages, keyword = topic_segment(request.question, chat_history)
            formatted_history = [f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in selected_messages]

            embedded_query = text_embedder.run(text=query)["embedding"]
            retrieved_docs = retriever.run(query_embedding=embedded_query)["documents"]
            wiki_docs = fetch_wikipedia_knowledge(keyword)

            all_docs_in_store = list(document_store.storage.values())
            doc_embeddings = {
                doc.id: doc.embedding for doc in all_docs_in_store if doc.embedding is not None
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
                selected_history=formatted_history,
                difficulty_desc=difficulty_desc,
                attention=attention
            )["prompt"]

            full_answer = ""
            for chunk in generator.generate_stream(prompt):
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

            memory.add_user_message(request.question)
            memory.add_assistant_message(full_answer)

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "content": f"处理请求失败：{str(e)}"
            }, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )

def generate_title_core(
    content: str,
    max_length: int = 6,
    min_length: int = 1
) -> str:

    payload_template = {
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.3,
        "top_p": 0.9,
        "think": False
    }
    prompt = f"""【Task】Generate a concise and accurate title for the input text.
    【Input Text】: {content}
    【Generation Rules】:
    1. The title must accurately summarize the core content of the text and remain faithful to the original theme.
    2. Return ONLY the title text without any additional explanation, punctuation, or formatting.
    【Output】:"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = payload_template.copy()
    payload["prompt"] = prompt

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        generated_title = result.get("response", "").strip()
        if not generated_title or generated_title.isspace():
            return ""
        return generated_title

    except Exception as e:
        print(f"Title generation failed：{str(e)}")
        return ""

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
    generated_title = generate_title_core(
        content=request.content,
        max_length=request.max_length,
        min_length=request.min_length
    )
    if generated_title:
        return {
            "success": True,
            "title": generated_title,
            "message": str(len(generated_title))
        }
    else:
        fallback_title = request.content[:request.max_length].strip()
        if len(request.content) > request.max_length:
            fallback_title += "..."
        return {
            "success": False,
            "title": fallback_title,
            "message": "Title generation failed"
        }

class DocumentLoadRequest(BaseModel):
    path: str


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

        # 处理文档嵌入并写入存储
        doc_embedder = pipeline_components["doc_embedder"]
        embedded_docs = doc_embedder.run(documents=docs)["documents"]
        document_store.write_documents(embedded_docs)

        return {
            "status": "success",
            "message": f"Successfully loaded and processed {len(docs)} document Fragment, from: {request.path}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process the document: {str(e)}"
        )



def generate_similar_math_question_core(
        question: str
) -> List[str]:
    payload_template = {
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.95,
        "think": False
    }

    prompt = f"""【Task】Based on the input math problem, generate 5 similar problems that strictly follow this difficulty distribution: 1 easy, 2 medium, and 2 harder problems.
    【Input Problem】: {question}
    【Difficulty Requirements】:
    1. Easy Problem: Remove 1-2 complex conditions, reduce the number of solution steps, and retain only the core knowledge points.
    2. Medium Problem: Keep the problem type, knowledge points, and solution steps consistent with the original problem; only replace numbers/scenarios.
    3. Harder Problem: Add 1 relevant knowledge point or constraint condition, resulting in more complex solution steps.
    【Output Requirements】:
    1. Output only the 5 problems, each on a new line. No category headings or numbering.
    2. Problems must be clearly stated and unambiguous, with reasonable number replacements.
    3. Must not contain any explanatory text or formatting marks.
    【Output】:"""

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = payload_template.copy()
    payload["prompt"] = prompt

    try:
        response = requests.post(url, json=payload, timeout=25)
        response.raise_for_status()
        result = response.json()
        generated_content = result.get("response", "").strip()

        similar_questions = [
            q.strip() for q in generated_content.split("\n")
            if q.strip() and not q.strip().startswith(("【", "1.", "2.", "3.", "4.", "5."))
        ][:5]

        return similar_questions

    except Exception as e:
        print(f"Similar API call for generating math problems failed：{str(e)}")
        return []


@app.post(
    "/generate-similar-math-question",
    response_model=SimilarMathQuestionResponse,
    summary="Generate similar math problems with fixed difficulty ratios",
    description="Receive the text of the math problem, and automatically generate 1 simple question, 2 questions of the same difficulty level, and 2 more difficult questions. Then return the question list directly.Receive the text of the math problem, and automatically generate 1 simple question, 2 questions of the same difficulty level, and 2 more difficult questions. Then return the question list directly."
)
def generate_similar_math_question(
        request: SimilarMathQuestionRequest,
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Math problems cannot be left blank.")

    similar_questions = generate_similar_math_question_core(request.question)
    total_count = len(similar_questions)

    if total_count == 5:
        return {
            "success": True,
            "similar_questions": similar_questions,
            "total_count": total_count,
            "message": "Successfully generated 5 similar questions that match the difficulty ratio."
        }
    else:
        return {
            "success": False,
            "similar_questions": similar_questions,
            "total_count": total_count,
            "message": f"expected 5 questions.Only {total_count} questions have been generated, which is less than the expected 5 questions."
        }

@app.post(
    "/recognize-content",
    response_model=OCRResponse,
    summary="Identify the content within the file（support /Word/PDF/TXT）",
    description="Upload image（jpg/png/jpeg）、Word（doc/docx）、PDF、TXTFile, return the recognized text contentFile, return the recognized text content"
)
async def recognize_content(
        file: UploadFile = File(..., description="support type：jpg、jpeg、png、doc、docx、pdf、txt")
) -> OCRResponse:
    supported_formats = {
        "jpg": "image",
        "jpeg": "image",
        "png": "image",
        "doc": "word",
        "docx": "word",
        "pdf": "pdf",
        "txt": "txt"
    }
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in supported_formats:
        supported = ", ".join(supported_formats.keys())
        raise HTTPException(
            status_code=400,
            detail=f"not support type：{file_ext}，only support{supported}"
        )
    file_type = supported_formats[file_ext]

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="file empty")

        result_text = ""
        if file_type == "image":
            with Image.open(io.BytesIO(file_bytes)).convert("L") as img:
                result_text = pytesseract.image_to_string(
                    img,
                    lang='chi_sim+eng+equ',
                    config='--psm 6'
                ).strip()

        elif file_type == "txt":
            result_text = file_bytes.decode("utf-8", errors="ignore").strip()

        elif file_type == "word":
            elements = partition(
                file=io.BytesIO(file_bytes),
                file_filename=file.filename,
                language="zh"
            )
            result_text = "\n".join(
                [elem.text.strip() for elem in elements if hasattr(elem, 'text') and elem.text.strip()])

        elif file_type == "pdf":
            try:
                elements = partition(
                    file=io.BytesIO(file_bytes),
                    file_filename=file.filename,
                    #这里是读中文的图，如果是英文题目需要切换"zh"到"en"
                    language="zh"
                )
                result_text = "\n".join(
                    [elem.text.strip() for elem in elements if hasattr(elem, 'text') and elem.text.strip()])
            except Exception as e:
                #print(f"unstructured解析PDF失败，尝试PyPDF2备选方案: {str(e)}")
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                result_text = "\n\n".join(text_parts)

        if not result_text:
            return OCRResponse(
                success=False,
                result="",
                message="No valid text was identified. Please ensure the file is clear and the content is complete (printable format is the best)."
            )

        return OCRResponse(
            success=True,
            result=result_text,
            message=f"{file_type.upper()}File identification successful"
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Tesseract-OCRNot found. Please confirm that the installation and configuration of the path have been done correctly."
        )
    except ImportError as e:
        missing_lib = str(e).split("'")[1] if "'" in str(e) else "Unknown dependency"
        raise HTTPException(
            status_code=500,
            detail=f"Lack of necessary dependent libraries：{missing_lib}，Please try again after installation"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Identification failure：{str(e)}"
        )


def find_leaf_nodes_with_history(data: dict) -> List[Tuple[str, int]]:
    """递归遍历树结构，收集所有有历史记录的叶子节点"""
    leaf_nodes = []

    def traverse(node: dict):
        if "children" not in node or not node["children"]:
            history_len = len(node.get("history", []))
            if history_len > 0:
                leaf_nodes.append((node["name"], history_len))
        else:
            for child in node["children"]:
                traverse(child)

    if isinstance(data, list):
        for item in data:
            traverse(item)
    elif isinstance(data, dict):
        traverse(data)

    return leaf_nodes


@app.get("/report_preference")
async def report_preference():
    """Get top 5 most frequently asked topics"""
    try:
        with open("tree.json", "r", encoding="utf-8") as f:
            tree_data = json.load(f)
        
        def find_leaf_nodes_with_history(node, path=""):
            """Recursively find all leaf nodes with history"""
            leaf_nodes = []
            current_path = f"{path}-{node['name']}" if path else node['name']
            
            if 'children' not in node or not node['children']:
                # ✅ 使用 history_records
                history_count = len(node.get('history_records', []))
                if history_count > 0:
                    leaf_nodes.append((current_path, history_count))
            else:
                for child in node['children']:
                    leaf_nodes.extend(find_leaf_nodes_with_history(child, current_path))
            
            return leaf_nodes
        
        all_leaf_nodes = find_leaf_nodes_with_history(tree_data)
        sorted_nodes = sorted(all_leaf_nodes, key=lambda x: (-x[1], x[0]))
        top_five = [node[0] for node in sorted_nodes[:5]]
        
        return JSONResponse(content=top_five)
    
    except FileNotFoundError:
        return JSONResponse(
            content={"error": "tree.json file not found"}, 
            status_code=404
        )
    except json.JSONDecodeError:
        return JSONResponse(
            content={"error": "Invalid JSON format in tree.json"}, 
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)