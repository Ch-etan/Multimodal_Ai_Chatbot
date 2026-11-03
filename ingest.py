import shutil, subprocess
from pathlib import Path
import cv2, pymupdf, ollama, pandas as pd
from docx import Document
from faster_whisper import WhisperModel
from config import *
import store

Path(DATA_DIR).mkdir(exist_ok=True)
MEDIA = Path(DATA_DIR) / "media"
MEDIA.mkdir(parents=True, exist_ok=True)
_whisper = None


def caption(path):
    try:
        r = ollama.chat(model=CHAT_MODEL, messages=[{
            "role": "user",
            "content": "Describe this image in detail. Transcribe any visible text exactly.",
            "images": [str(path)]}],
            options={"num_ctx": NUM_CTX}, keep_alive=KEEP_ALIVE)
        return r["message"]["content"]
    except Exception:
        return ""


def split(text):
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + CHUNK_CHARS].strip()
        if piece:
            out.append(piece)
        if i + CHUNK_CHARS >= len(text):
            break
        i += CHUNK_CHARS - CHUNK_OVERLAP
    return out


# ---- loaders: each yields (kind, payload, meta); kind = text | chunk | image ----

def load_text(p):
    yield "text", p.read_text(errors="ignore"), {}

def load_image(p):
    yield "image", p, {}

def load_pdf(p):
    doc = pymupdf.open(p)
    for i, page in enumerate(doc, 1):
        txt = page.get_text().strip()
        if len(txt) < 30:  # scanned page -> let the vision model read it
            f = MEDIA / f"{p.stem}_p{i}.png"
            page.get_pixmap(dpi=150).save(f)
            yield "image", f, {"page": i}
        else:
            yield "text", txt, {"page": i}
        for n, img in enumerate(page.get_images(full=True)):
            info = doc.extract_image(img[0])
            if min(info["width"], info["height"]) < 120:
                continue
            f = MEDIA / f"{p.stem}_p{i}_{n}.{info['ext']}"
            f.write_bytes(info["image"])
            yield "image", f, {"page": i}

def load_docx(p):
    if p.suffix.lower() == ".doc":  # legacy format: needs LibreOffice
        subprocess.run(["soffice", "--headless", "--convert-to", "docx", "--outdir", str(MEDIA), str(p)], check=True)
        p = MEDIA / (p.stem + ".docx")
    d = Document(p)
    yield "text", "\n".join(x.text for x in d.paragraphs if x.text.strip()), {}
    for t in d.tables:
        rows = [" | ".join(c.text.strip() for c in r.cells) for r in t.rows]
        yield "chunk", "Table:\n" + "\n".join(rows), {}
    for n, rel in enumerate(r for r in d.part.rels.values() if "image" in r.reltype):
        f = MEDIA / f"{p.stem}_{n}{Path(rel.target_ref).suffix or '.png'}"
        f.write_bytes(rel.target_part.blob)
        yield "image", f, {}

def load_table(p):
    if p.suffix.lower() == ".csv":
        sheets = {"csv": pd.read_csv(p)}
    else:
        sheets = pd.read_excel(p, sheet_name=None)
    for name, df in sheets.items():
        df = df.dropna(how="all")
        stats = df.describe(include="all").T.head(30).to_string()
        yield "chunk", f"Table '{name}': {len(df)} rows. Columns: {', '.join(map(str, df.columns))}.\n{stats}", {"sheet": str(name)}
        for s in range(0, len(df), 25):
            batch = df.iloc[s:s + 25].to_csv(index=False)
            yield "chunk", f"Sheet {name}, rows {s + 1}-{s + 25}:\n{batch}", {"sheet": str(name)}

def load_audio(p):
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_SIZE, device=WHISPER_DEVICE, compute_type="int8", cpu_threads=WHISPER_THREADS)
    segs, _ = _whisper.transcribe(str(p), vad_filter=True)
    buf, start = [], None
    for s in segs:
        start = s.start if start is None else start
        buf.append(s.text.strip())
        if s.end - start >= 60:
            yield "chunk", "Transcript: " + " ".join(buf), {"start": int(start)}
            buf, start = [], None
    if buf:
        yield "chunk", "Transcript: " + " ".join(buf), {"start": int(start)}

def load_video(p):
    try:
        yield from load_audio(p)
    except Exception:
        pass  # no audio track
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fno in range(0, total, max(1, int(fps * FRAME_EVERY_SEC))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            break
        t = int(fno / fps)
        f = MEDIA / f"{p.stem}_t{t}.jpg"
        cv2.imwrite(str(f), frame)
        yield "image", f, {"start": t}

LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".doc": load_docx,
           ".csv": load_table, ".xlsx": load_table, ".xls": load_table,
           ".txt": load_text, ".md": load_text,
           **{e: load_image for e in IMG_EXT},
           **{e: load_audio for e in (".mp3", ".wav", ".m4a", ".flac", ".ogg")},
           **{e: load_video for e in (".mp4", ".mov", ".mkv", ".avi", ".webm")}}


def ingest_file(path):
    src = Path(path)
    p = Path(DATA_DIR) / src.name
    if src.resolve() != p.resolve():
        shutil.copy(src, p)
    loader = LOADERS.get(p.suffix.lower())
    if not loader:
        return f"Unsupported file type: {p.name}"
    store.delete_source(p.name)
    items, n_img = [], 0
    for kind, payload, meta in loader(p):
        meta = {"source": p.name, **meta}
        if kind == "text":
            items += [(c, meta) for c in split(payload)]
        elif kind == "chunk":
            items.append((payload, meta))
        elif kind == "image":
            if store.add_image(payload, meta):
                n_img += 1
                cap = caption(payload) if CAPTION_IMAGES else ""
                if cap:
                    items.append((f"[Image description] {cap}", {**meta, "image": str(payload)}))
    store.add_chunks(items)
    return f"{p.name}: {len(items)} text chunks, {n_img} images"
