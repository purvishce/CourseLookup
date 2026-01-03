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

    # -------- Overview Chunk --------
    overview_text = f"""
                    [COURSE OVERVIEW]
                    Title: {course.title}
                    Instructor: {course.instructor}
                    Level: {course.level}
                    Rating: {course.rating} out of 5 based on {course.reviewcount} reviews
                    Duration: {course.duration}
                    Lectures: {course.lectures}
                    """.strip()

    chunks.append({
        "text": overview_text,
        "metadata": {**metadata, "chunk_type": "overview"}
    })

    # -------- Description Chunk --------
    description_text = f"""
                    [COURSE DESCRIPTION]
                    {course.description}
                    """.strip()

    if len(course.description) > 0:
        chunks.append({
            "text": description_text,
            "metadata": {**metadata, "chunk_type": "description"}
        })
        
     # -------- Metadata-only Chunk --------
    metadata_text = f"""
                [COURSE METADATA]
                Title: {course.title}
                Instructor: {course.instructor}
                Level: {course.level}
                Rating: {course.rating}
                ReviewCount: {course.reviewcount}
                Duration: {course.duration}
                Lectures: {course.lectures}
                """.strip()
    chunks.append({
        "text": metadata_text,
        "metadata": {**metadata, "chunk_type": "metadata"}
    })
    print(chunks)

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


def embed_texts(texts, EMBEDDING_MODEL="text-embedding-3-large"):
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