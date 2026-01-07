import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from chromadb import PersistentClient
from models import Chunk, CourseRecord
from openai import OpenAI
import os

def build_metadata(course: CourseRecord) -> dict:
    return {
        "title": course.title,
        "instructor": course.instructor,
        "level": course.level,
        "rating": course.rating,
        "reviewcount": course.reviewcount,
        "duration": course.duration,
        "lectures": course.lectures
    }

def create_chunks(course: CourseRecord) -> list[dict]:
    metadata = build_metadata(course)
    chunks = []

    # Build a simple set of keyword tags from title, instructor and description
    import re

    def tokens_from(text: str, limit: int = 50) -> list[str]:
        words = re.findall(r"[A-Za-z0-9]+", text.lower())
        words = [w for w in words if len(w) > 2]
        seen = []
        for w in words:
            if w not in seen:
                seen.append(w)
            if len(seen) >= limit:
                break
        return seen

    title_tokens = tokens_from(course.title)
    instr_tokens = tokens_from(course.instructor)
    desc_tokens = tokens_from(course.description or "")

    tags = list(dict.fromkeys(title_tokens + instr_tokens + desc_tokens))

    # -------- Combined Chunk --------
    combined_text = f"""
                    Title: {course.title}
                    Instructor: {course.instructor}
                    Instructor: {course.instructor} courses
                    Instructor: {course.instructor} Udemy
                    Level: {course.level}
                    Rating: {course.rating}
                    ReviewCount: {course.reviewcount}
                    Duration: {course.duration}
                    Lectures: {course.lectures}
                    Description: {course.description}
                    """.strip()

    chunks.append({
        "text": combined_text,
        "metadata": {**metadata, "chunk_type": "combined", "tags": ",".join(tags)}
    })

    return chunks
"""
def detect_role(turn: str) -> str:
    text = turn.lower()

    if "customer" in text:
        return "customer"
    if "agent" in text or "support" in text:
        return "agent"

    return "unknown"

def classify_turn_type(turn: str) -> str:
    text = turn.lower()

    if any(x in text for x in ["resolved", "fixed", "issue resolved"]):
        return "outcome_success"

    if any(x in text for x in ["did not work", "issue persists", "still happening"]):
        return "outcome_failure"

    if any(x in text for x in ["try", "please", "suggest", "recommend"]):
        return "recommendation"

    return "statement"


def create_chunks1(email):
    turns = split_by_turns(email.full_text)
    metadata = build_metadata(email)
    chunks = []

    # Simple grouping: first customer, then agent, then resolution
    customer_turns, agent_turns, resolution_turns = [], [], []

    for turn in turns:
        if "customer" in turn.lower():
            customer_turns.append(turn)
        elif "agent" in turn.lower():
            agent_turns.append(turn)
        else:
            resolution_turns.append(turn)

    if customer_turns:
        chunks.append(Chunk(text="\n".join(customer_turns), metadata=metadata))
    if agent_turns:
        chunks.append(Chunk(text="\n".join(agent_turns), metadata=metadata))
    if resolution_turns:
        chunks.append(Chunk(text="\n".join(resolution_turns), metadata=metadata))

    return chunks
"""


def embed_texts(texts, EMBEDDING_MODEL="text-embedding-3-small"):
    """
    Generate embeddings for a list of texts
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    embeddings = []
    batch_size = 1000

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        embeddings.extend([item.embedding for item in response.data])

    return embeddings