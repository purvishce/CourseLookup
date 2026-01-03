# Course Lookup using RAG

A course advisor assistant that uses vector search and AI to recommend Udemy courses based on user queries.

## Features

- Ingests course data from Udemy CSV
- Creates text chunks with metadata
- Generates embeddings using OpenAI
- Stores in ChromaDB vector database
- Provides a Gradio-based chat interface for querying courses
- Supports metadata filters (instructor, rating, level)

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see installation)

## Installation

1. Clone or download the repository.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If no requirements.txt, install manually:
   ```bash
   pip install openai langchain langchain-openai langchain-chroma chromadb gradio python-dotenv tqdm litellm numpy scikit-learn pydantic
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Setup

1. Place your Udemy courses CSV file as `Udemy.csv` in the project root. The CSV should have columns: title, description, instructor, rating, reviewcount, duration, lectures, level.

2. Run the ingestion script to load data into the vector database:
   ```bash
   python ingest.py
   ```

   This will create chunks, generate embeddings, and store them in the `vector_db` directory.

## Usage

Run the retriever script to start the Gradio interface:
```bash
python retirver.py
```

Open the provided local URL (usually http://127.0.0.1:7860) in your browser to interact with the course advisor.

Ask questions like:
- "Recommend Python courses by John Doe"
- "Show me beginner level courses with 4.5+ rating"
- "What courses are available on machine learning?"

## Project Structure

- `ingest.py`: Loads courses, creates chunks, generates embeddings, stores in DB
- `retirver.py`: Gradio interface for querying courses
- `helperfunction.py`: Utility functions for chunking and metadata
- `models.py`: Data models (CourseRecord, Chunk)
- `Udemy.csv`: Input course data
- `vector_db/`: ChromaDB persistent storage
- `evaluation/`: Evaluation scripts
- `implementation/`: Additional implementation files

## Notes

- Ensure your OpenAI API key has sufficient credits for embeddings and chat.
- The system uses GPT-4.1-nano for responses.
- Filters are extracted from queries using simple regex; can be improved with NLP.</content>
<parameter name="filePath">c:\DEV\Projects\CustomerServiceLookup\README.md
