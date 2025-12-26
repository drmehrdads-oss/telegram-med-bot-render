import os
import logging
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Build the Telegram application (no polling!)
tg_app: Application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I’m your medical assistant bot.\n\n"
        "Ask a health question and I’ll help (not a doctor)."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send a message like: 'I have a headache and fever, what could it be?'.")


async def medical_assistant_reply(user_text: str) -> str:
    system = (
        "You are a helpful medical assistant. You provide general health information, "
        "not a diagnosis. Encourage seeking professional care when appropriate. "
        "If symptoms sound urgent (chest pain, trouble breathing, stroke signs, severe bleeding, etc.), "
        "advise emergency services."
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    await update.message.chat.send_action(action="typing")

    try:
        answer = await medical_assistant_reply(user_text)
    except Exception:
        logger.exception("OpenAI error")
        answer = "Sorry — I hit an error generating a reply. Please try again."

    await update.message.reply_text(answer)


tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(CommandHandler("help", help_cmd))
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))


@app.on_event("startup")
async def on_startup():
    await tg_app.initialize()
    await tg_app.start()
    logger.info("Telegram application started (webhook mode).")


@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()
    logger.info("Telegram application stopped.")


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
