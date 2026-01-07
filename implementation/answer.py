from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
import re
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 50

llm = ChatOpenAI(model=MODEL)

def expand_query(query: str) -> str:
    prompt = f"Expand this search query for Udemy courses with related keywords, topics, and popular instructor names. Keep it concise. Original: {query}"
    response = llm.invoke([HumanMessage(content=prompt)])
    expanded = response.content.strip()
    return f"{query} {expanded}"

SYSTEM_PROMPT = """
You are a professional Course Advisor Assistant.

You answer user questions strictly based on the provided course catalog context retrieved from the knowledge base.
The context consists of structured course information, including:
- Course title
- Description
- Instructor(s)
- Level (e.g., Beginner, All Levels)
- Rating and number of reviews
- Duration and number of lectures

Guidelines:
- Use ONLY the information present in the context.
- If multiple courses are relevant, synthesize a clear and concise response.
- Do NOT invent courses, instructors, or course details.
- If the context does not contain enough information to answer, clearly say that the information is not available.
- Do NOT reference internal IDs, row numbers, or retrieval mechanics.
- Maintain a polite, professional, and educational tone.
- Summarize information clearly using bullet points when applicable.
- Highlight course recommendations based on relevance, ratings, or user-specified criteria when appropriate.

Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, collection_name="customerlookup_emails", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 100})  # initial fetch for reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')


def extract_filters_from_query(query: str):
    """
    Detect metadata filters from user query
    """
    filters = {}

    # Detect instructor (simple regex, can improve)
    instructor_match = re.findall(r"by ([A-Za-z\s\.]+)", query, re.IGNORECASE)
    if instructor_match:
        filters["instructor"] = instructor_match[0].strip()

    # Detect rating (5-star, 4-star, etc.)
    rating_match = re.findall(r"(\d(?:\.\d)?)\s*(?:star|rating)", query, re.IGNORECASE)
    if rating_match:
        filters["rating"] = {"$gte": float(rating_match[0])}

    # Detect level (beginner, all levels, advanced)
    levels = ["beginner", "all levels", "intermediate", "advanced"]
    for level in levels:
        if level in query.lower():
            filters["level"] = level.capitalize()

    return filters


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question, with filters.
    """
    filters = extract_filters_from_query(question)
    question = expand_query(question)
    # initial retrieve more for reranking
    search_kwargs = {"k": 300}
    if filters:
        search_kwargs["filter"] = filters
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)

    # rerank with cross-encoder and keep top RETRIEVAL_K
    if docs and len(docs) > RETRIEVAL_K:
        sentences = [doc.page_content for doc in docs]
        pairs = [(question, s) for s in sentences]
        try:
            scores = cross_encoder.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            docs = [docs[i] for i in sorted_indices[:RETRIEVAL_K]]
        except Exception:
            # fallback to original ordering if reranker fails
            docs = docs[:RETRIEVAL_K]

    return docs


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
