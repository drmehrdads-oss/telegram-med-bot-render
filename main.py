import os
import io
import json
import logging
from typing import Optional

from fastapi import FastAPI, Request, Response

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# Optional PDF support (recommended)
# pip install pymupdf
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medbot")


# ----------------------------
# ENV
# ----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # vision+text capable
PUBLIC_URL = os.getenv("PUBLIC_URL")  # e.g. https://telegram-med-bot-render.onrender.com
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "telegram-webhook")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")
if not PUBLIC_URL:
    raise RuntimeError("Missing PUBLIC_URL env var (your Render service URL)")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
tg_app: Optional[Application] = None


# ----------------------------
# MEDICAL SYSTEM PROMPT (Persian)
# ----------------------------
SYSTEM_PROMPT_FA = """\
تو یک دستیار اطلاعات پزشکی به زبان فارسی هستی (پزشک نیستی).
وظیفه: با توجه به علائم کاربر، 1) هشدار اورژانسی بده اگر لازم است،
2) چند احتمال کلی را توضیح بده (بدون تشخیص قطعی)،
3) آزمایش‌های پیشنهادی (Blood/Urine) و تصویربرداری مناسب (X-ray/US/CT/MRI) را پیشنهاد بده
   و برای هر کدام توضیح کوتاه بده «چرا» و «چه زمانی لازم است»،
4) اقدامات ساده و ایمن در خانه (اگر مناسب است)،
5) چه زمانی باید پزشک/اورژانس مراجعه کند.

قواعد ایمنی:
- هرگز دارو تجویز نکن (دوز/نام داروی نسخه‌ای نده). فقط توصیه‌های عمومی ایمن.
- اگر علائم خطر (درد قفسه سینه، تنگی نفس شدید، ضعف یکطرفه، گیجی، خونریزی شدید، غش، تب بالا همراه با سفتی گردن...) وجود دارد:
  تاکید کن که احتمال اورژانسی است و باید فوراً به اورژانس/پزشک مراجعه کند.
- اگر کاربر آزمایش/عکس ارسال کرد: فقط یافته‌های غیرطبیعی/پرچم قرمز را مشخص کن،
  توضیح بده ممکن است زمینه‌های مختلف داشته باشد و نیاز به تفسیر پزشک دارد.

قالب پاسخ:
- 🚨 هشدار اورژانسی (اگر لازم است)
- 🔎 خلاصه وضعیت (1-2 خط)
- 🧪 آزمایش‌های پیشنهادی
- 🩻 تصویربرداری پیشنهادی
- ✅ کارهایی که الان می‌توان انجام داد
- 🧭 سوال‌های تکمیلی (حداکثر 5 سوال)
"""


def _fa_intro() -> str:
    return (
        "سلام 👋 من دستیار اطلاعات پزشکی هستم (پزشک نیستم).\n"
        "علائم‌تان را + سن + سابقه بیماری/داروها را بنویسید.\n"
        "می‌توانید نتیجه آزمایش، عکس (JPEG/PNG) یا PDF گزارش را هم ارسال کنید.\n"
        "اگر وضعیت اورژانسی است همین الان با اورژانس تماس بگیرید."
    )


# ----------------------------
# OpenAI helper
# ----------------------------
def openai_text_answer(user_text: str) -> str:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_FA},
            {"role": "user", "content": user_text},
        ],
    )
    return resp.output_text.strip()


def openai_image_answer(user_text: str, image_bytes: bytes, mime: str) -> str:
    # OpenAI vision input
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_FA},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_data": image_bytes, "mime_type": mime},
                ],
            },
        ],
    )
    return resp.output_text.strip()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF. If fitz isn't installed or extraction fails, return "".
    """
    if fitz is None:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for i in range(min(doc.page_count, 5)):  # first 5 pages is usually enough
            page = doc.load_page(i)
            t = page.get_text("text") or ""
            t = t.strip()
            if t:
                parts.append(f"[صفحه {i+1}]\n{t}")
        return "\n\n".join(parts).strip()
    except Exception as e:
        logger.exception("PDF text extraction failed: %s", e)
        return ""


def render_pdf_first_page_png(pdf_bytes: bytes) -> bytes:
    """
    Render first page of PDF to PNG (for scanned PDFs).
    """
    if fitz is None:
        return b""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    # Increase resolution a bit
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


# ----------------------------
# Telegram handlers
# ----------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(_fa_intro())


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    # Quick Persian instruction for "tests suggestion"
    # We let the model do the real work via the system prompt.
    answer = openai_text_answer(user_text)
    await update.message.reply_text(answer)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Get the highest resolution photo
    photos = update.message.photo or []
    if not photos:
        return
    photo = photos[-1]

    file = await context.bot.get_file(photo.file_id)
    bio = io.BytesIO()
    await file.download_to_memory(out=bio)
    image_bytes = bio.getvalue()

    caption = (update.message.caption or "").strip()
    user_text = (
        "این یک تصویر از نتیجه آزمایش/گزارش/عکس پزشکی است. "
        "لطفاً موارد غیرطبیعی یا پرچم قرمز را مشخص کن و اگر لازم است آزمایش یا تصویربرداری پیشنهادی بده.\n"
        f"توضیحات کاربر: {caption or 'ندارد'}"
    )

    answer = openai_image_answer(user_text, image_bytes=image_bytes, mime="image/jpeg")
    await update.message.reply_text(answer)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if not doc:
        return

    # Allow PDFs and images in documents too
    mime = (doc.mime_type or "").lower()
    file = await context.bot.get_file(doc.file_id)
    bio = io.BytesIO()
    await file.download_to_memory(out=bio)
    data = bio.getvalue()

    caption = (update.message.caption or "").strip()

    # If PDF: try text extraction; if empty -> render first page -> vision
    if mime == "application/pdf" or (doc.file_name or "").lower().endswith(".pdf"):
        text = extract_pdf_text(data)
        if text:
            user_text = (
                "این متن استخراج‌شده از PDF گزارش/آزمایش است. "
                "فقط موارد غیرطبیعی یا مهم را مشخص کن و سپس بر اساس علائم احتمالی، آزمایش‌ها و تصویربرداری مناسب را پیشنهاد بده. "
                "دارو تجویز نکن.\n\n"
                f"توضیحات کاربر: {caption or 'ندارد'}\n\n"
                f"متن PDF:\n{text}"
            )
            answer = openai_text_answer(user_text)
            await update.message.reply_text(answer)
            return

        # scanned PDF fallback: render first page and use vision
        if fitz is None:
            await update.message.reply_text(
                "برای خواندن PDF لازم است روی Render پکیج PyMuPDF نصب باشد (pymupdf). "
                "فعلاً نمی‌توانم PDF را پردازش کنم. لطفاً از محتوای گزارش عکس واضح (JPEG/PNG) بفرستید."
            )
            return

        png = render_pdf_first_page_png(data)
        user_text = (
            "این تصویر صفحه اول PDF است (احتمالاً اسکن شده). "
            "لطفاً متن/اعداد را بخوان و موارد غیرطبیعی یا مهم را مشخص کن و در صورت نیاز آزمایش‌ها/تصویربرداری پیشنهادی بده. "
            "دارو تجویز نکن.\n"
            f"توضیحات کاربر: {caption or 'ندارد'}"
        )
        answer = openai_image_answer(user_text, image_bytes=png, mime="image/png")
        await update.message.reply_text(answer)
        return

    # If image as a document (jpeg/png)
    if mime in ("image/jpeg", "image/png") or (doc.file_name or "").lower().endswith((".jpg", ".jpeg", ".png")):
        use_mime = "image/png" if mime == "image/png" or (doc.file_name or "").lower().endswith(".png") else "image/jpeg"
        user_text = (
            "این یک تصویر از نتیجه آزمایش/گزارش/عکس پزشکی است. "
            "لطفاً موارد غیرطبیعی یا پرچم قرمز را مشخص کن و اگر لازم است آزمایش یا تصویربرداری پیشنهادی بده.\n"
            f"توضیحات کاربر: {caption or 'ندارد'}"
        )
        answer = openai_image_answer(user_text, image_bytes=data, mime=use_mime)
        await update.message.reply_text(answer)
        return

    await update.message.reply_text(
        "فعلاً فقط PDF و عکس (JPEG/PNG) را می‌توانم بررسی کنم. "
        "اگر گزارش دارید، لطفاً PDF یا عکس واضح ارسال کنید."
    )


# ----------------------------
# FastAPI webhook bridge
# ----------------------------
@app.get("/health")
async def health():
    return {"ok": True}


@app.post(f"/telegram/{WEBHOOK_SECRET}")
async def telegram_webhook(req: Request):
    if tg_app is None:
        return Response(status_code=503, content="Bot not ready")

    data = await req.json()
    update = Update.de_json(data, tg_app.bot)

    # Process update
    await tg_app.process_update(update)
    return {"ok": True}


@app.on_event("startup")
async def on_startup():
    global tg_app
    tg_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    tg_app.add_handler(CommandHandler("start", cmd_start))
    tg_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    tg_app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    await tg_app.initialize()
    await tg_app.start()

    webhook_url = f"{PUBLIC_URL}/telegram/{WEBHOOK_SECRET}"
    await tg_app.bot.set_webhook(url=webhook_url)

    logger.info("Telegram webhook set to: %s", webhook_url)


@app.on_event("shutdown")
async def on_shutdown():
    global tg_app
    if tg_app is not None:
        await tg_app.stop()
        await tg_app.shutdown()
        tg_app = None
