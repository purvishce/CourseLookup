from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import re
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
from typing import List


# -------------------------
# Load environment variables
# -------------------------
load_dotenv(override=True)

# -------------------------
# Models & DB
# -------------------------
MODEL = "gpt-4.1-nano"
DB_NAME = "vector_db"
collection_name = "customerlookup_emails"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load or create Chroma vectorstore
vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings,
    collection_name=collection_name
)


# Retriever & LLM
llm = ChatOpenAI(temperature=0, model_name=MODEL)


# -------------------------
# System prompt template
# -------------------------
SYSTEM_PROMPT_TEMPLATE = """
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


# -------------------------
# Retriever with optional metadata filters
# -------------------------
def retrieve_courses(query: str, top_k=15):
    """
    Retrieve relevant courses from vectorstore
    """
    filters = extract_filters_from_query(query)
    search_kwargs = {"k": top_k}
    if filters:
        search_kwargs["filter"] = filters
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


# -------------------------
# Answer function
# -------------------------
def answer_question(question: str, history=[]):
    docs = retrieve_courses(question)

    if not docs:
        return "Sorry, no relevant courses were found in the catalog."

    # Combine retrieved chunks into context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Format system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    # Generate LLM response
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content

# -------------------------
# Gradio interface
# -------------------------
gr.ChatInterface(answer_question).launch()