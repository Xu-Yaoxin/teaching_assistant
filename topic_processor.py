import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from config import OLLAMA_BASE_URL, MODEL_NAME


def topic_segment(query, messages):

    payload_template = {
        "model": MODEL_NAME,
        "stream": False,
        "temperature": 0.2,
        "top_p": 0.9,
        "think": False  
    }

    class SimilarityModel:
        def __init__(self):

            self.model = SentenceTransformer("all-MiniLM-L12-v2")

        def calculate_similarity(self, sentence1: str, sentence2: str) -> float:

            embedding1 = self.model.encode(sentence1)
            embedding2 = self.model.encode(sentence2)
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return similarity

    def process_dialogue(model, query, messages, similarity_threshold=0):


        result_messages = []
        for msg in messages[:-1]:
            user = msg.get("user", "").strip()
            robot = msg.get("assistant", "").strip()
            #s1 = f"{user} {robot}"
            s1 = user
            s2 = query.strip()
            similarity = model.calculate_similarity(s1, s2)
            if similarity >= similarity_threshold:
                is_continuous = call_api(s1, s2)
                if is_continuous:
                    result_messages.append({"user": user, "assistant": robot})
        if messages:
            last_msg = messages[-1]
            result_messages.append(
                {"user": last_msg.get("user", "").strip(), "assistant": last_msg.get("assistant", "").strip()})
        return result_messages

    def call_api(dialogue1, dialogue2):

        url = f"{OLLAMA_BASE_URL}/api/generate"
        prompt_content = f"""Determine whether the following two dialogues belong to the same topic (no task objective/subject switch). Answer only "yes" or "no".
dialogue1:
{dialogue1}
dialogue2:
{dialogue2}
"""

        payload = payload_template.copy()
        payload["prompt"] = prompt_content
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            return "Yes" in answer
        except Exception as e:
            print(f"API call failed: {e}")
            return False

    def rewrite(query, history_messages):

        url = f"{OLLAMA_BASE_URL}/api/generate"
        prompt = f"""
【Task】:Based on historical conversation data, rewrite the current user's query to make it more complete and clear, ensuring it can be understood without relying on past conversations.
【Input】：
1. Historical Dialogue Records
{history_messages}
2. Current user's query (content requiring rewriting):
{query}
【Rewriting Requirements】
1. Preserve the core intent of the current query without altering the user's original request.
2. Extract necessary context from historical conversations and supplement it into the current query to resolve ambiguities.
3. Use concise and natural language, avoiding redundancy and omitting irrelevant content from past dialogues.
4. The rewritten query must be self-contained and understandable: it should convey the user's intended meaning completely, even without knowledge of the historical dialogue.
5. You only need to output the rewritten query; no other fields are required.
【Output】
"""
        payload = payload_template.copy()
        payload["prompt"] = prompt
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            print("Rewritten Query: ", answer or query)
            return answer or query
        except Exception as e:
            print(f"rewriteAPI call failed: {e}")
            return query

    def extract_keyword(query):

        url = f"{OLLAMA_BASE_URL}/api/generate"
        prompt = f"""
【Task Description】：You are a professional linguist. You now need to extract the single keyword most relevant to the given sentence from the input sentences.
【Input】Input sentence：{query}
【Requeirements】
1. You may only return the most relevant keyword.
2. You need only return the keyword; no other fields are required.
        """
        payload = payload_template.copy()
        payload["prompt"] = prompt
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            print("Search Keywords:", answer or query)
            return answer or query
        except Exception as e:
            print(f"extract_keywordAPI call fail: {e}")
            return query

    similarity_model = SimilarityModel()
    result_messages = process_dialogue(similarity_model, query, messages)
    rewritten_query = rewrite(query, result_messages)
    keyword = extract_keyword(rewritten_query)

    return rewritten_query, result_messages, keyword