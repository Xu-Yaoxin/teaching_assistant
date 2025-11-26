import hashlib
import json
from pathlib import Path
import wikipedia
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Text
from haystack.dataclasses import Document


def fetch_wikipedia_knowledge(query: str, lang: str = "En") -> list[Document]:

    hash_value = hashlib.md5(f"{query}_{lang}".encode()).hexdigest()
    cache_path = Path(f"wiki_cache/{hash_value}.json")

    if cache_path.exists():
        #print(f"✅ Load Wikipedia content from the cache: {query}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return [Document(**d) for d in json.load(f)]

    wikipedia.set_lang(lang)
    try:
        page = wikipedia.page(query, auto_suggest=True)
        elements = [Text(text=page.content)]
        chunks = chunk_by_title(elements)

        docs = []
        for i, chunk in enumerate(chunks):
            content = chunk.text.strip()
            if content:
                print(f"Content：{content[:25]}..." if len(content) > 25 else f"Content：{content}")
                doc = Document(
                    content=content,
                    meta={
                        "source": "wikipedia",
                        "title": page.title,
                        "url": page.url,
                        "type": "WIKIPEDIA",
                        "chunk_id": i + 1,
                        "total_chunks": len(chunks),
                        "original_length": len(page.content)
                    }
                )
                docs.append(doc)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(
                [{"content": d.content, "meta": d.meta} for d in docs],
                f,
                ensure_ascii=False,
                indent=2
            )

        return docs

    except wikipedia.exceptions.PageError:
        #print(f"❌ No entry related to '{query}' was found on Wikipedia")
        return []
    except wikipedia.exceptions.DisambiguationError as e:
        #print(f"⚠️ Ambiguous entry, try to obtain candidate entries: {e.options[:3]}")
        docs = []

        for option in e.options[:5]:
            try:
                sub_page = wikipedia.page(option)
                elements = [Text(text=sub_page.content)]
                chunks = chunk_by_title(elements)

                for i, chunk in enumerate(chunks):
                    content = chunk.text.strip()
                    if content:
                        print(f"Content：{content[:25]}..." if len(content) > 25 else f"Content：{content}")
                        docs.append(Document(
                            content=content,
                            meta={
                                "source": "wikipedia",
                                "title": sub_page.title,
                                "url": sub_page.url,
                                "type": "WIKIPEDIA",
                                "chunk_id": i + 1,
                                "total_chunks": len(chunks),
                                "disambiguation": True
                            }
                        ))
            except:
                continue

        if docs:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [{"content": d.content, "meta": d.meta} for d in docs],
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        return docs
    except Exception as e:
        print(f"❌ The Wikipedia API call failed：{str(e)}")
        return []