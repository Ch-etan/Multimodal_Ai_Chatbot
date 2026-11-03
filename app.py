from pathlib import Path
import gradio as gr
import ollama
from config import *
import store, ingest

APP_NAME = "Nexa AI"

SYSTEM = (f"You are {APP_NAME}, a helpful multimodal assistant. Answer using ONLY the provided context "
          "(text, transcripts, image descriptions and the attached images). Cite sources as [1], [2]. "
          "If the answer is not in the context, say you don't know.")

CSS = """
.nexa-header{background:linear-gradient(135deg,#4f46e5,#06b6d4);padding:22px 28px;border-radius:16px;margin-bottom:12px}
.nexa-header h1{margin:0;font-size:2.1rem;color:#fff !important}
.nexa-header p{margin:4px 0 0;color:#fff !important;opacity:.92}
.nexa-badges span{display:inline-block;background:rgba(255,255,255,.2);color:#fff;border-radius:999px;padding:3px 12px;margin:12px 6px 0 0;font-size:.8rem}
footer{display:none !important}
"""

HEADER = f"""<style>{CSS}</style>
<div class="nexa-header">
  <h1>{APP_NAME}</h1>
  <p>Chat with your documents, spreadsheets, images, audio and video. 100% local and private.</p>
  <div class="nexa-badges"><span>PDF</span><span>Word</span><span>Excel / CSV</span><span>Images</span><span>Audio</span><span>Video</span><span>Runs offline</span></div>
</div>"""

EXAMPLES = [{"text": "Summarize everything in the knowledge base.", "files": []},
            {"text": "What are the key points mentioned in the audio or video?", "files": []},
            {"text": "What text appears in the images?", "files": []}]


def label(m):
    s = m["source"]
    if "page" in m: s += f", p.{m['page']}"
    if "sheet" in m: s += f", sheet {m['sheet']}"
    if "start" in m: s += f", @{m['start']}s"
    return s


def retrieve(q, query_imgs):
    docs, metas, imgs = [], [], list(query_imgs)
    if store.texts.count() and q:
        r = store.texts.query(query_embeddings=store.embed_query(q),
                              n_results=min(TOP_K_TEXT, store.texts.count()))
        docs, metas = r["documents"][0], r["metadatas"][0]
        for m in metas:  # images linked through their captions
            if "image" in m and m["image"] not in imgs:
                imgs.append(m["image"])
    if query_imgs and store.images.count():  # image-to-image similarity via CLIP
        r = store.images.query(query_embeddings=store.clip_images(query_imgs),
                               n_results=min(TOP_K_IMG, store.images.count()))
        for ms in r["metadatas"]:
            for m in ms:
                if m["path"] not in imgs:
                    imgs.append(m["path"])
                    docs.append(f"[Similar indexed image from {label(m)}]")
                    metas.append(m)
    return docs, metas, imgs[:TOP_K_IMG + len(query_imgs)]


def chat(message, history):
    q = (message.get("text") or "").strip()
    query_imgs = [f for f in message.get("files", []) if Path(f).suffix.lower() in IMG_EXT]
    if not q and query_imgs:
        q = "Describe this image and relate it to the knowledge base."
    docs, metas, imgs = retrieve(q, query_imgs)
    ctx = "\n\n".join(f"[{i}] ({label(m)})\n{d}" for i, (d, m) in enumerate(zip(docs, metas), 1))
    msgs = [{"role": "system", "content": SYSTEM}]
    for h in history[-6:]:
        if isinstance(h.get("content"), str):
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": f"Context:\n{ctx or '(nothing indexed yet)'}\n\nQuestion: {q}",
                 "images": imgs})
    out = ""
    for part in ollama.chat(model=CHAT_MODEL, messages=msgs, stream=True,
                            options={"num_ctx": NUM_CTX}, keep_alive=KEEP_ALIVE):
        out += part["message"]["content"]
        yield out
    if metas:
        out += "\n\n**Sources**\n" + "\n".join(f"{i}. {label(m)}" for i, m in enumerate(metas, 1))
        yield out


def library():
    names = set()
    for c in (store.texts, store.images):
        names |= {m["source"] for m in c.get(include=["metadatas"])["metadatas"]}
    return "\n".join(f"- {n}" for n in sorted(names)) or "_Nothing indexed yet_"


def index(files, progress=gr.Progress()):
    log = []
    for f in progress.tqdm(files or [], desc="Indexing"):
        try:
            log.append(ingest.ingest_file(f))
        except Exception as e:
            log.append(f"{Path(f).name}: FAILED ({e})")
    return ("\n".join(log) or "No files selected."), library()


def clear():
    store.reset()
    return "Knowledge base cleared.", library()


def chat_kwargs():
    return dict(
        multimodal=True,
        chatbot=gr.Chatbot(height=540, placeholder=f"### Hi, I'm {APP_NAME}\nIndex some files on the left, then ask me anything."),
        textbox=gr.MultimodalTextbox(file_types=["image"], placeholder=f"Ask {APP_NAME} about your files... (attach an image to search with it)"),
    )


with gr.Blocks(title=APP_NAME) as demo:
    gr.HTML(HEADER)
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(file_count="multiple", label="1. Add knowledge files")
            with gr.Row():
                btn = gr.Button("Index files", variant="primary")
                clr = gr.Button("Clear all", variant="secondary")
            status = gr.Textbox(label="Status", lines=4)
            with gr.Accordion("Indexed files", open=True):
                lib = gr.Markdown(library())
            btn.click(index, files, [status, lib])
            clr.click(clear, None, [status, lib])
        with gr.Column(scale=3):
            try:
                gr.ChatInterface(chat, examples=EXAMPLES, cache_examples=False, **chat_kwargs())
            except Exception:
                gr.ChatInterface(chat, **chat_kwargs())

if __name__ == "__main__":
    demo.launch()
