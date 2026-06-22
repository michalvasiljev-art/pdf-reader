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
import shutil
import concurrent.futures
from collections import Counter

import fitz
import edge_tts
import pygame
from deep_translator import GoogleTranslator

# Tesseract на Windows часто не попадает в PATH текущего процесса
_TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR"
if os.path.isdir(_TESSERACT_WIN) and _TESSERACT_WIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _TESSERACT_WIN + os.pathsep + os.environ.get("PATH", "")

# ─── Настройки ────────────────────────────────────────────────────────────────

VOICE         = "ru-RU-SvetlanaNeural"
RATE          = "+0%"
CHUNK         = 800
MARGIN_TOP    = 0.13
MARGIN_BOTTOM = 0.08

paused      = False
should_stop = False

# ─── Регулярки ────────────────────────────────────────────────────────────────

URL_RE           = re.compile(r'https?://\S+|www\.\S+|\S+\.\S+/\S*', re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r'^\s*[\*†‡§¶]|^(Примечание|Источник|Note|Source)\b', re.IGNORECASE)
ENDS_PUNCT_RE    = re.compile(r'[.!?…]$')
SENTENCE_BREAK   = re.compile(r'(?<=[.!?…])\s+')


# ══════════════════════════════════════════════════════════════════════════════
# TextTranslator — определение языка и перевод на русский
# ══════════════════════════════════════════════════════════════════════════════

LANG_NAMES = {
    'en': 'английского', 'de': 'немецкого', 'fr': 'французского',
    'es': 'испанского',  'it': 'итальянского', 'zh-cn': 'китайского',
    'ja': 'японского',   'ko': 'корейского',   'pt': 'португальского',
    'pl': 'польского',   'uk': 'украинского',
}


class TextTranslator:
    """Определяет язык документа и переводит текст на русский."""

    def __init__(self):
        self.source_lang: str = 'ru'
        self._active: bool    = False

    def detect(self, sample: str) -> str:
        """Определяет язык по доле кириллицы. Не-русский → Google auto-detect."""
        cyrillic = sum(1 for c in sample if 'Ѐ' <= c <= 'ӿ')
        ratio = cyrillic / max(len(sample), 1)
        if ratio > 0.25:
            self.source_lang = 'ru'
            self._active = False
        else:
            self.source_lang = 'auto'
            self._active = True
        return self.source_lang

    @property
    def needs_translation(self) -> bool:
        return self._active

    @property
    def lang_label(self) -> str:
        if self.source_lang == 'auto':
            return 'авто-определение'
        return LANG_NAMES.get(self.source_lang, self.source_lang)

    def translate(self, text: str) -> str:
        """Переводит текст на русский. При ошибке возвращает оригинал."""
        if not self._active or not text.strip():
            return text
        try:
            result = GoogleTranslator(source=self.source_lang, target='ru').translate(text)
            return result or text
        except Exception:
            return text


# ══════════════════════════════════════════════════════════════════════════════
# PageExtractor — всё что касается извлечения и фильтрации текста
# ══════════════════════════════════════════════════════════════════════════════

class PageExtractor:
    """Извлекает чистый текст из страниц PDF, готовый для TTS.

    Для обычных PDF: блочный разбор с фильтрами (колонтитулы, таблицы, ссылки).
    Для сканов: OCR через Tesseract (если установлен).
    Исправляет переносы слов. Определяет заголовки по размеру шрифта.
    """

    HEADING_RATIO  = 1.15   # шрифт >= body_size * HEADING_RATIO → заголовок
    SCAN_TEXT_MAX  = 50     # символов — меньше этого = страница считается сканом

    def __init__(self, doc, start_idx: int = 0):
        self._repeated    = self._scan_repeated_headers(doc, start_idx)
        self._body_size   = self._detect_body_size(doc, start_idx)
        self.ocr_available = shutil.which('tesseract') is not None

    # ── публичный метод ───────────────────────────────────────────────────────

    def process(self, page) -> list:
        """Возвращает список (kind, text): kind = 'heading' | 'body'."""
        # Скан: мало текста, есть изображения — пускаем через OCR
        if self.ocr_available and self._is_scanned(page):
            return self._ocr_page(page)

        h      = page.rect.height
        top    = h * MARGIN_TOP
        bottom = h * (1 - MARGIN_BOTTOM)

        trects = self._table_rects(page)
        lrects = [fitz.Rect(lnk["from"]) for lnk in page.get_links() if lnk.get("from")]

        segments = []
        for b in page.get_text("dict").get("blocks", []):
            if b.get("type") != 0:
                continue
            _, y0, _, y1 = b["bbox"]
            if y0 < top or y1 > bottom:
                continue

            b_rect = fitz.Rect(b["bbox"])
            if trects and any(b_rect.intersects(r) for r in trects):
                continue
            if lrects and any(b_rect.intersects(r) for r in lrects):
                continue

            text, is_heading = self._parse_block(b)
            if not text:
                continue
            if text in self._repeated:
                continue
            if re.fullmatch(r'\d+', text):
                continue
            if TABLE_CAPTION_RE.match(text):
                continue

            text = URL_RE.sub('', text).strip()
            if not text:
                continue

            # Пауза TTS после заголовка — добавляем точку если нет знака препинания
            if is_heading and not ENDS_PUNCT_RE.search(text):
                text += '.'

            segments.append(('heading' if is_heading else 'body', text))

        return segments

    # ── приватные методы ──────────────────────────────────────────────────────

    @staticmethod
    def _scan_repeated_headers(doc, start_idx: int, sample: int = 30, threshold: float = 0.35) -> set:
        """Находит текст, повторяющийся в зонах колонтитулов."""
        counter = Counter()
        pages = list(doc)[start_idx: start_idx + sample]
        n = len(pages)
        if not n:
            return set()
        for page in pages:
            h = page.rect.height
            for b in page.get_text("blocks"):
                if b[6] != 0:
                    continue
                if b[1] / h < 0.15 or b[3] / h > 0.85:
                    counter[b[4].strip()] += 1
        return {t for t, c in counter.items() if t and c / n >= threshold}

    @staticmethod
    def _detect_body_size(doc, start_idx: int, sample: int = 10) -> float:
        """Определяет основной размер шрифта (самый частый по символам)."""
        sizes = Counter()
        for page in list(doc)[start_idx: start_idx + sample]:
            for b in page.get_text("dict").get("blocks", []):
                if b.get("type") != 0:
                    continue
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        t = span["text"].strip()
                        if t:
                            sizes[round(span["size"], 1)] += len(t)
        return sizes.most_common(1)[0][0] if sizes else 11.0

    @staticmethod
    def _table_rects(page) -> list:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    lambda: [fitz.Rect(t.bbox) for t in page.find_tables().tables]
                )
                return future.result(timeout=1.0)
        except Exception:
            return []

    def _parse_block(self, block) -> tuple:
        """Возвращает (text, is_heading). Склеивает переносы слов."""
        raw_lines = []
        sizes = []

        for line in block.get("lines", []):
            line_text = "".join(s["text"] for s in line.get("spans", []))
            raw_lines.append(line_text)
            for span in line.get("spans", []):
                if span["text"].strip():
                    sizes.append(span["size"])

        text = self._fix_hyphens("\n".join(raw_lines))

        avg_size   = sum(sizes) / len(sizes) if sizes else self._body_size
        is_heading = avg_size >= self._body_size * self.HEADING_RATIO

        return text, is_heading

    def _is_scanned(self, page) -> bool:
        """True если страница — скан: мало текста, но есть изображения."""
        text   = page.get_text().strip()
        images = page.get_images(full=False)
        return len(images) > 0 and len(text) < self.SCAN_TEXT_MAX

    def _ocr_page(self, page) -> list:
        """OCR скана через Tesseract (встроен в PyMuPDF). Возвращает сегменты."""
        try:
            tp  = page.get_textpage_ocr(language='rus+eng', dpi=300, full=False)
            raw = page.get_text(textpage=tp)
        except Exception:
            return []

        text = self._fix_hyphens(raw)
        if not text:
            return []

        segments = []
        for line in text.splitlines():
            line = line.strip()
            if not line or re.fullmatch(r'\d+', line):
                continue
            # Короткая строка без знака препинания в конце — вероятно заголовок
            if len(line) < 80 and not ENDS_PUNCT_RE.search(line) and line[0].isupper():
                segments.append(('heading', line + '.'))
            else:
                segments.append(('body', line))
        return segments

    @staticmethod
    def _fix_hyphens(raw: str) -> str:
        """Убирает переносы слов и склеивает строки."""
        raw = raw.replace('\xad', '')  # мягкий перенос (U+00AD)
        # перенос: буква-\nбуква → склеиваем без дефиса
        raw = re.sub(r'([а-яёА-ЯЁa-zA-Z])-\n([а-яёА-ЯЁa-zA-Z])', r'\1\2', raw)
        # остальные переводы строк → пробел
        raw = re.sub(r'\n+', ' ', raw)
        return raw.strip()


# ─── TTS ──────────────────────────────────────────────────────────────────────

async def generate_audio(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data += chunk["data"]
    return data


def play_mp3_bytes(data: bytes) -> None:
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


# ─── Текст → чанки ────────────────────────────────────────────────────────────

def split_chunks(text: str, max_chars: int = CHUNK) -> list:
    """Делит текст на куски до max_chars, предпочитая границы предложений."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    chunks = []
    while len(text) > max_chars:
        # ищем последнюю границу предложения внутри лимита
        boundary = -1
        for m in SENTENCE_BREAK.finditer(text[:max_chars]):
            boundary = m.end()
        if boundary <= 0:
            boundary = text[:max_chars].rfind(' ')
        if boundary <= 0:
            boundary = max_chars
        chunks.append(text[:boundary].strip())
        text = text[boundary:].strip()

    if text:
        chunks.append(text)
    return chunks


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    global should_stop

    if len(sys.argv) < 2:
        print("Использование:  python reader.py <файл.pdf>")
        print("Опции:          --голос DmitryNeural  --скорость +20%  --page 5  --debug")
        sys.exit(1)

    args       = sys.argv[1:]
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

    start_idx = max(0, start_page - 1)

    if debug:
        doc = fitz.open(pdf_path)
        print(f"\n=== DEBUG: блоки стр. {start_page}–{start_page + 2} ===")
        for pg_i, page in enumerate(list(doc)[start_idx: start_idx + 3]):
            h = page.rect.height
            print(f"\n--- Страница {start_idx + pg_i + 1}  (высота={h:.0f}) ---")
            for b in page.get_text("blocks"):
                if b[6] != 0:
                    continue
                preview = b[4].strip().replace("\n", " ")[:60]
                print(f"  y={b[1]:.0f}-{b[3]:.0f}  ({b[1]/h:.2f}-{b[3]/h:.2f})  '{preview}'")
        sys.exit(0)

    print(f"\n📖  {os.path.basename(pdf_path)}")
    doc       = fitz.open(pdf_path)
    total     = len(doc)
    extractor = PageExtractor(doc, start_idx)

    # Определяем язык по первым абзацам начальной страницы
    translator = TextTranslator()
    sample_segments = extractor.process(doc[start_idx])
    sample_text = " ".join(t for _, t in sample_segments[:5])
    if sample_text.strip():
        lang = translator.detect(sample_text)
        lang_info = f"перевод с {translator.lang_label}" if translator.needs_translation else "русский"
    else:
        lang_info = "—"

    ocr_status = "✓ Tesseract" if extractor.ocr_available else "✗ Tesseract не найден (сканы пропускаются)"
    print(f"    всего стр.: {total}  |  старт: стр. {start_page}  |  язык: {lang_info}")
    print(f"    голос: {VOICE}  |  скорость: {RATE}  |  OCR: {ocr_status}")
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
            segments = await loop.run_in_executor(None, extractor.process, doc[pdf_idx])
            for _kind, text in segments:
                if should_stop:
                    break
                if translator.needs_translation:
                    text = await loop.run_in_executor(None, translator.translate, text)
                if not text:
                    continue
                for chunk in split_chunks(text):
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
