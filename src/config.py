import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROS_DISTRO = "humble"
DOCS_DIR = Path("data/ros2_documentation/source")
INCLUDE_DIRS = ["Tutorials", "How-To-Guides", "Concepts", "Installation"]
DOCS_BASE_URL = f"https://docs.ros.org/en/{ROS_DISTRO}"

INDEX_PATH = Path("data/index.faiss")
CHUNKS_PATH = Path("data/chunks.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_CHARS = 1200      # chunk size
OVERLAP_CHARS = 200   # small text blocks carried into next chunk
TOP_K = 5

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
