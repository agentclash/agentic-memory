import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
MEDIA_STORAGE_PATH = os.getenv("MEDIA_STORAGE_PATH", "./data/media")
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 768  # Matryoshka truncation — can bump to 1536 later
MEDIA_EMBED_MAX_BYTES = int(os.getenv("MEDIA_EMBED_MAX_BYTES", str(20 * 1024 * 1024)))
