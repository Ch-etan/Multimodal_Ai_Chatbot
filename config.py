CHAT_MODEL = "qwen2.5vl:7b"        # any Ollama vision model (llama3.2-vision, gemma3:4b, minicpm-v ...)
TEXT_EMBED = "BAAI/bge-small-en-v1.5"
IMAGE_EMBED = "clip-ViT-B-32"
WHISPER_SIZE = "small"             # tiny / base / small / medium
DB_DIR = "db"
DATA_DIR = "data"
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200
FRAME_EVERY_SEC = 10
CAPTION_IMAGES = True              # set False for faster (but less searchable) ingestion
TOP_K_TEXT = 6
TOP_K_IMG = 2                      # fewer images per prompt = less VRAM
# --- 8 GB VRAM profile: keep the GPU for the vision LLM only ---
EMBED_DEVICE = "cpu"               # bge + CLIP are small; CPU is fine
WHISPER_DEVICE = "cpu"             # int8 on the i7-13700HX is fast enough
WHISPER_THREADS = 8
NUM_CTX = 4096                     # context window; raise to 6144 only if VRAM allows
KEEP_ALIVE = "10m"                 # unload the LLM after 10 idle minutes
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
