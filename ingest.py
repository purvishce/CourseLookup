import sys
import os  # Added missing import for os.getenv
from pathlib import Path

from chromadb import PersistentClient

from helperfunction import create_chunks, embed_texts

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from tqdm import tqdm
from litellm import completion
import numpy as np
from sklearn.manifold import TSNE
#import plotly.graph_objects as go  # Commented out as per previous fix; install plotly if needed
import csv
from datetime import datetime
from models import CourseRecord

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"

DB_NAME = "vector_db"
collection_name = "customerlookup_emails"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path("knowledge-base")
#AVERAGE_CHUNK_SIZE = 500

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

openai = OpenAI()

# Define csv_path (assuming it's in the project root or adjust as needed)
csv_path = Path("Udemy.csv")  # Changed to relative path; adjust if the file is in a different location


print("📥 Loading courses...")
# Read csv file and scan through each row
courses = []
# Try multiple encodings
encodings = ['utf-8-sig', 'cp1252', 'utf-16', 'iso-8859-1', 'latin1']
for enc in encodings:
    try:
        with open(csv_path, newline='', encoding=enc) as f:
            reader = csv.DictReader(f)
            for row in reader:
                courses.append(CourseRecord(                    
                    title=row['title'],
                    description=row.get('description', ''),
                    instructor=row['instructor'],
                    rating=float(row['rating']) if row['rating'] else 0.0,
                    reviewcount=int(row['reviewcount']) if row['reviewcount'] else 0,
                    duration=row['duration'],
                    lectures=row['lectures'],
                    level=row['level']
                    
                ))
        break  # Success, exit loop
    except (UnicodeDecodeError, UnicodeError):
        continue  # Try next encoding

print(len(courses))

#next step is to create chunks from emails
print("✂️ Creating chunks...")
chunks = []
for course in courses:
      chunks.extend(create_chunks(course))
    
print(f"✅ Total chunks created: {len(chunks)}")
#print(chunks[0])

print("🧠 Generating embeddings...")
texts = []
texts = [chunk["text"] for chunk in chunks]
print(f"Total texts to embed: {len(texts)}")

chroma = PersistentClient(path=DB_NAME)
if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

# Batch the embeddings to avoid token limit
vectors = []
batch_size = 1000  # Adjust batch size as needed to stay under token limits
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    emb = openai.embeddings.create(model=embedding_model, input=batch).data
    vectors.extend([e.embedding for e in emb])

collection = chroma.get_or_create_collection(name=collection_name)

ids = [str(i) for i in range(len(chunks))]
metas = [chunk["metadata"] for chunk in chunks]

# Batch the add operation to avoid ChromaDB batch size limit
batch_size = 1000  # Adjust as needed
for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i + batch_size]
    batch_vectors = vectors[i:i + batch_size]
    batch_texts = texts[i:i + batch_size]
    batch_metas = metas[i:i + batch_size]
    collection.add(ids=batch_ids, embeddings=batch_vectors, documents=batch_texts, metadatas=batch_metas)

print(f"Vectorstore created with {collection.count()} documents")

dimensions = len(vectors[0]) if vectors else 0

count = collection.count()
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

print("✅ Ingestion complete.")