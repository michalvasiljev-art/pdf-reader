#!/usr/bin/env python3
"""PDF Reader — читает русскоязычные PDF вслух через нейронный TTS Microsoft."""

import sys
import os
import asyncio
import time
import threading
import tempfile
import msvcrt

import re
import concurrent.futures
from collections import Counter

import fitz          # PyMuPDF
import edge_tts
import pygame

VOICE        = "ru-RU-SvetlanaNeural"   # fem: SvetlanaNeural, male: DmitryNeural
RATE         = "+0%"                     # скорость: +20% быстрее, -20% медленнее
CHUNK        = 800                       # символов за один TTS-запрос
MARGIN_TOP    = 0.13                     # доля высоты страницы — зона колонтитула сверху
MARGIN_BOTTOM = 0.08                     # доля высоты страницы — зона колонтитула снизу

paused      = False
should_stop = False


# ─── PDF ──────────────────────────────────────────────────────────────────────

def find_repeated_headers(doc, start_idx: int = 0, sample: int = 30, threshold: float = 0.35):
    """Возвращает множество текстов, которые повторяются в зонах колонтитулов."""
    counter = Counter()
    pages = list(doc)[start_idx: start_idx + sample]
    n = len(pages)
    if n == 0:
        return set()
    for page in pages:
        h = page.rect.height
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            rel_top = b[1] / h
            rel_bot = b[3] / h
            if rel_top < 0.15 or rel_bot > 0.85:
                counter[b[4].strip()] += 1
    return {t for t, c in counter.items() if t and c / n >= threshold}


URL_RE          = re.compile(r'https?://\S+|www\.\S+|\S+\.\S+/\S*', re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r'^\s*[\*†‡§¶]|^(Примечание|Источник|Note|Source)\b', re.IGNORECASE)


def overlapping_rects(b_rect, rects):
    return any(b_rect.intersects(r) for r in rects)


def get_link_rects(page):
    return [fitz.Rect(lnk["from"]) for lnk in page.get_links() if lnk.get("from")]


def get_table_rects(page):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(lambda: [fitz.Rect(t.bbox) for t in page.find_tables().tables])
            return future.result(timeout=1.0)
    except Exception:
        return []


def process_page(page, repeated):
    """Извлекает текст одной страницы с фильтрацией."""
    h       = page.rect.height
    top     = h * MARGIN_TOP
    bottom  = h * (1 - MARGIN_BOTTOM)
    lrects  = get_link_rects(page)
    trects  = get_table_rects(page)

    blocks = []
    for b in page.get_text("blocks"):
        if b[6] != 0:
            continue
        if b[1] < top or b[3] > bottom:
            continue
        text = b[4].strip()
        if not text:
            continue
        if text in repeated:
            continue
        if re.fullmatch(r'\d+', text):
            continue
        b_rect = fitz.Rect(b[:4])
        if lrects and overlapping_rects(b_rect, lrects):
            continue
        if trects and overlapping_rects(b_rect, trects):
            continue
        if TABLE_CAPTION_RE.match(text):
            continue
        text = URL_RE.sub('', text).strip()
        if text:
            blocks.append(text)

    return "\n".join(blocks).strip()


def split_chunks(text: str, max_chars: int = CHUNK):
    """Делит текст на куски по абзацам, не превышая max_chars."""
    chunks, current = [], ""
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = (current + " " + para) if current else para
    if current:
        chunks.append(current)
    return chunks


# ─── TTS ──────────────────────────────────────────────────────────────────────

async def generate_audio(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data += chunk["data"]
    return data


def play_mp3_bytes(data: bytes) -> None:
    """Сохраняет mp3 во временный файл и воспроизводит через pygame."""
    global should_stop
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        pygame.mixer.music.load(tmp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not should_stop:
            time.sleep(0.05)
        pygame.mixer.music.stop()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ─── Клавиатура ───────────────────────────────────────────────────────────────

def key_listener() -> None:
    global paused, should_stop
    while not should_stop:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b" ":
                paused = not paused
                if paused:
                    pygame.mixer.music.pause()
                    print("\n⏸  Пауза  (пробел — продолжить, Q — выход)")
                else:
                    pygame.mixer.music.unpause()
                    print("\n▶  Продолжаю...")
            elif key in (b"q", b"Q"):
                should_stop = True
                pygame.mixer.music.stop()
                print("\n⏹  Остановлено.")
        time.sleep(0.05)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    global should_stop

    if len(sys.argv) < 2:
        print("Использование:  python reader.py <файл.pdf>")
        print("Опции:          --голос DmitryNeural  --скорость +20%  --page 5")
        sys.exit(1)

    # Разбор аргументов
    args = sys.argv[1:]
    pdf_path   = None
    start_page = 1
    debug      = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--голос" and i + 1 < len(args):
            globals()["VOICE"] = f"ru-RU-{args[i+1]}"
            i += 2
        elif a == "--скорость" and i + 1 < len(args):
            globals()["RATE"] = args[i + 1]
            i += 2
        elif a == "--page" and i + 1 < len(args):
            start_page = int(args[i + 1])
            i += 2
        elif a == "--debug":
            debug = True
            i += 1
        else:
            if not a.startswith("--"):
                pdf_path = a
            i += 1

    if not pdf_path or not os.path.exists(pdf_path):
        print(f"Файл не найден: {pdf_path}")
        sys.exit(1)

    if debug:
        start_idx = max(0, start_page - 1)
        doc = fitz.open(pdf_path)
        print(f"\n=== DEBUG: блоки стр. {start_page}–{start_page+2} ===")
        for pg_i, page in enumerate(list(doc)[start_idx: start_idx + 3]):
            h = page.rect.height
            print(f"\n--- Страница {start_idx + pg_i + 1}  (высота={h:.0f}) ---")
            for b in page.get_text("blocks"):
                if b[6] != 0:
                    continue
                rel_top = b[1] / h
                rel_bot = b[3] / h
                preview = b[4].strip().replace("\n", " ")[:60]
                print(f"  y={b[1]:.0f}-{b[3]:.0f}  ({rel_top:.2f}-{rel_bot:.2f})  '{preview}'")
        sys.exit(0)

    print(f"\n📖  {os.path.basename(pdf_path)}")
    start_idx = max(0, start_page - 1)

    doc      = fitz.open(pdf_path)
    total    = len(doc)
    repeated = find_repeated_headers(doc, start_idx=start_idx)

    print(f"    всего стр.: {total}  |  старт: стр. {start_page}  |  голос: {VOICE}  |  скорость: {RATE}")
    print("    Пробел — пауза/продолжить    Q — выход\n")

    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.mixer.init()

    threading.Thread(target=key_listener, daemon=True).start()

    queue = asyncio.Queue(maxsize=2)
    loop  = asyncio.get_event_loop()

    async def producer():
        for pdf_idx in range(start_idx, total):
            if should_stop:
                break
            page      = doc[pdf_idx]
            page_text = await loop.run_in_executor(None, process_page, page, repeated)
            for chunk in split_chunks(page_text):
                if should_stop:
                    break
                audio = await generate_audio(chunk)
                await queue.put((pdf_idx, audio))
        await queue.put(None)

    async def consumer():
        current_page = -1
        while True:
            item = await queue.get()
            if item is None or should_stop:
                break
            pdf_idx, audio = item
            if pdf_idx != current_page:
                current_page = pdf_idx
                print(f"  [стр. {pdf_idx + 1}/{total}]")
            while paused and not should_stop:
                await asyncio.sleep(0.1)
            if should_stop:
                break
            await loop.run_in_executor(None, play_mp3_bytes, audio)

    await asyncio.gather(producer(), consumer())

    if not should_stop:
        print("\n✓  Чтение завершено.")

    pygame.mixer.quit()


if __name__ == "__main__":
    asyncio.run(main())
