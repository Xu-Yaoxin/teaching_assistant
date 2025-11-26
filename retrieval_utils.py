import numpy as np
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.dataclasses import Document


def filter_docs_by_similarity(
        docs: list[Document],
        query_embedding: list[float],
        embedder: SentenceTransformersTextEmbedder,
        doc_embeddings: dict = None,
        threshold: float = 0.3
) -> list[Document]:
    filtered = []
    for doc in docs:
        if doc_embeddings and doc.id in doc_embeddings:
            doc_embedding = doc_embeddings[doc.id]
        else:
            doc_embedding = embedder.run(text=doc.content)["embedding"]

        similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        doc.meta["similarity_score"] = round(similarity, 4)

        if similarity >= threshold:
            filtered.append(doc)

    return filtered