import hashlib
import json
from pathlib import Path
import PyPDF2
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Text, ElementMetadata

# Compatibility import for different versions of Haystack
try:
    from haystack.dataclasses import Document
except ImportError:
    from haystack import Document

# Lazy import for whisper
def import_whisper():
    try:
        import whisper
        return whisper
    except ImportError:
        return None
    except Exception as e:
        print(f"Failed to import whisper: {str(e)}")
        return None

def transcribe_video(video_path: str) -> list[Document]:
    """Convert video to timestamped text segments with caching mechanism"""
    hash_value = hashlib.md5(open(video_path, 'rb').read()).hexdigest()
    cache_path = Path(f"video_cache/{hash_value}.json")

    if cache_path.exists():
        print(f"✅ Loaded transcription from cache: {Path(video_path).name}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        return [Document(**seg) for seg in segments]

    model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
    result = model.transcribe(video_path, verbose=False)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        {
            "content": seg["text"].strip(),
            "meta": {
                "source": Path(video_path).name,
                "file_type": "VIDEO",
                "start_time": seg["start"],
                "end_time": seg["end"],
                "duration": seg["end"] - seg["start"]
            }
        }
        for seg in result["segments"]
    ]
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    return [Document(**seg) for seg in segments]

def load_documents(source_path):
    """Load files from the given path (supporting both single files and directories), return a list of Documents"""
    path = Path(source_path)
    all_docs = []

    if path.is_file():
        if path.suffix.lower() in [".mp4", ".avi", ".mov"]:
            file_docs = transcribe_video(str(path))
        else:
            file_docs = _process_file(path)
        all_docs.extend(file_docs)

    elif path.is_dir():
        for file_path in path.glob("*.*"):
            suffix = file_path.suffix.lower()
            if suffix in [".mp4", ".avi", ".mov"]:
                file_docs = transcribe_video(str(file_path))
            elif suffix in [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt"]:
                file_docs = _process_file(file_path)
            else:
                print(f"⚠️ Skipping unsupported file type: {file_path.name}")
                continue
            all_docs.extend(file_docs)

    return all_docs

def _process_file(file_path):
    """Process a single non-video file (PDF/Word/PPT/TXT), only supports text-based PDFs"""
    try:
        suffix = file_path.suffix.lower()
        type_map = {
            ".pdf": "PDF",
            ".docx": "WORD",
            ".doc": "WORD",
            ".pptx": "PPT",
            ".ppt": "PPT",
            ".txt": "TXT"
        }
        file_type = type_map.get(suffix, suffix[1:].upper())

        try:
            elements = partition(str(file_path), language="zh")
        except Exception as e:
            if file_type == "PDF":
                print(f"⚠️ Failed to parse PDF with unstructured, trying PyPDF2 alternative: {file_path.name} - Error: {str(e)}")
                from unstructured.documents.elements import Text, ElementMetadata
                elements = []
                try:
                    import PyPDF2
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page_num, page in enumerate(reader.pages, 1):
                            text = page.extract_text()
                            if text and text.strip():  # Only keep non-empty text
                                # Use the correct ElementMetadata format
                                meta = ElementMetadata()
                                meta.page_number = page_num  # Set page number
                                elements.append(Text(text=text.strip(), metadata=meta))
                except Exception as pdf_err:
                    print(f"⚠️ Failed to parse PDF with PyPDF2: {file_path.name} - Error: {str(pdf_err)}")
                    return []
            else:
                raise 

        chunks = chunk_by_title(elements)

        documents = []
        for i, chunk in enumerate(chunks):
            content = chunk.text.strip()
            if not content:
                continue

            meta = {
                "source": file_path.name,
                "file_type": file_type,
                "chunk_id": i + 1,
                "total_chunks": len(chunks),
                "full_path": str(file_path)
            }

            if file_type == "PDF":
                page_num = getattr(chunk.metadata, "page_number", None)
                if page_num:
                    meta["page"] = page_num

            if file_type == "PPT":
                slide_num = getattr(chunk.metadata, "slide_number", None)
                if slide_num:
                    meta["slide"] = slide_num
                    
            if file_type == "WORD":
                paragraph_num = getattr(chunk.metadata, "paragraph_number", None)
                if paragraph_num:
                    meta["paragraph"] = paragraph_num

            documents.append(Document(content=content, meta=meta))

        print(f"✅ Loaded {len(documents)} segments from: {file_path.name}")
        return documents

    except Exception as e:
        print(f"⚠️ Failed to process file: {file_path.name} - Error: {str(e)}")
        return []

def format_documents_as_code(path: str) -> str:
    """Format document list as Python code string (for verification)"""
    docs = load_documents(path)
    print(f"\nThe document contains {len(docs)} content segments")
    formatted = "[\n"
    for doc in docs:
        escaped_content = doc.content.replace('"', '\\"').replace('\n', '\\n')
        formatted += f'    Document(content="{escaped_content}", meta={doc.meta}),\n'
    formatted += "]"
    return formatted
