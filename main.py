import os
import re
import io
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# PDF / image helpers
import pdfplumber
from PIL import Image

# Optional OCR
OCR_AVAILABLE = False
try:
    import pytesseract  # type: ignore
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# -------------------------
# ENV
# -------------------------
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -------------------------
# Profiles storage (persistent)
# -------------------------
PROFILES_FILE = "profiles.json"

def load_profiles() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(PROFILES_FILE):
        return {}
    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_profiles(profiles: Dict[str, Dict[str, Any]]) -> None:
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

PROFILES: Dict[str, Dict[str, Any]] = load_profiles()

def user_key(update: Update) -> str:
    # Use Telegram user id as key
    return str(update.effective_user.id)

def get_profile(update: Update) -> Dict[str, Any]:
    return PROFILES.get(user_key(update), {})

def set_profile(update: Update, age: int, sex: str) -> None:
    PROFILES[user_key(update)] = {"age": age, "sex": sex}
    save_profiles(PROFILES)

def reset_profile(update: Update) -> None:
    k = user_key(update)
    if k in PROFILES:
        del PROFILES[k]
        save_profiles(PROFILES)


# -------------------------
# Persian-only system prompt
# -------------------------
SYSTEM_PROMPT = (
    "شما یک دستیار اطلاعات پزشکی هستید (پزشک نیستید). "
    "همیشه فقط به زبان فارسی پاسخ بده. "
    "تشخیص قطعی نده. "
    "کار اصلی شما این است که نتایج آزمایش/ادرار/گزارش تصویربرداری را از روی متن بخوانی و موارد خارج از محدوده را مشخص کنی. "
    "اگر محدوده مرجع (Reference Range) روی برگه وجود دارد، فقط با همان مقایسه کن. "
    "اگر محدوده مرجع وجود ندارد، بگو بدون محدوده مرجع نمی‌توان با اطمینان قضاوت کرد و پیشنهاد کن محدوده/سن/جنس را بدهند. "
    "برای تصویربرداری (مثل X-ray) فقط خلاصه یافته‌ها و جملات نگران‌کننده را مشخص کن و توصیه کن با پزشک بررسی شود. "
    "اگر علائم اورژانسی یا عبارات خطرناک دیدی (مثل درد قفسه سینه، تنگی نفس شدید، خونریزی شدید، افت هوشیاری)، "
    "به‌صورت واضح توصیه کن فوراً با اورژانس تماس بگیرند یا به اورژانس مراجعه کنند."
)


# -------------------------
# Helpers: numeric/range
# -------------------------
def parse_float(x: str) -> Optional[float]:
    if x is None:
        return None
    x = x.strip().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", x)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def range_to_tuple(rng: str) -> Tuple[Optional[float], Optional[float]]:
    if not rng:
        return (None, None)
    s = rng.strip()

    m = re.search(r"^\s*<\s*([0-9]+(\.[0-9]+)?)\s*$", s)
    if m:
        return (None, float(m.group(1)))
    m = re.search(r"^\s*<=\s*([0-9]+(\.[0-9]+)?)\s*$", s)
    if m:
        return (None, float(m.group(1)))
    m = re.search(r"^\s*>\s*([0-9]+(\.[0-9]+)?)\s*$", s)
    if m:
        return (float(m.group(1)), None)
    m = re.search(r"^\s*>=\s*([0-9]+(\.[0-9]+)?)\s*$", s)
    if m:
        return (float(m.group(1)), None)

    s2 = s.replace("–", "-").replace("—", "-")
    m = re.search(r"([0-9]+(\.[0-9]+)?)\s*-\s*([0-9]+(\.[0-9]+)?)", s2)
    if m:
        return (float(m.group(1)), float(m.group(3)))

    return (None, None)

def classify(value: float, low: Optional[float], high: Optional[float]) -> str:
    if low is not None and value < low:
        return "پایین"
    if high is not None and value > high:
        return "بالا"
    return "نرمال"


# -------------------------
# Extract text from PDF/image
# -------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n".join(text_parts).strip()

def extract_text_from_image(image_bytes: bytes) -> str:
    if not OCR_AVAILABLE:
        return ""
    img = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(img)


# -------------------------
# OpenAI: structure extraction
# -------------------------
def openai_extract_structured(report_text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    schema_hint = {
        "type": "lab|urine|imaging|unknown",
        "items": [
            {"name": "string", "value": "string", "unit": "string", "range": "string", "flag": "string"}
        ],
        "imaging_summary": "string",
        "imaging_red_flags": ["string"],
        "notes": "string"
    }

    profile_text = ""
    if profile:
        profile_text = f"پروفایل کاربر: سن={profile.get('age')}، جنسیت={profile.get('sex')}.\n"

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    profile_text
                    + "این متن مربوط به نتیجه آزمایش/ادرار/تصویربرداری است.\n"
                    "لطفاً آن را به JSON مطابق این شِما تبدیل کن و فقط JSON خروجی بده:\n"
                    f"{json.dumps(schema_hint, ensure_ascii=False)}\n\n"
                    "متن گزارش:\n"
                    f"{report_text}"
                ),
            },
        ],
    )

    raw = (resp.output_text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def build_flag_report(data: Dict[str, Any], profile: Dict[str, Any]) -> str:
    r_type = (data.get("type") or "unknown").strip().lower()

    profile_line = ""
    if profile:
        profile_line = f"👤 پروفایل: {profile.get('sex')}، {profile.get('age')} ساله\n\n"

    if r_type == "imaging":
        summary = (data.get("imaging_summary") or "").strip()
        red_flags = data.get("imaging_red_flags") or []
        notes = (data.get("notes") or "").strip()

        msg = profile_line + "📌 **خلاصه گزارش تصویربرداری (اطلاعاتی):**\n"
        msg += (summary or "متن کافی برای خلاصه‌سازی پیدا نشد.") + "\n\n"

        if red_flags:
            msg += "⚠️ **نکات/عبارات قابل توجه:**\n"
            for x in red_flags[:10]:
                msg += f"• {x}\n"
            msg += "\n"

        msg += "✅ نتیجه را با پزشک/رادیولوژیست بررسی کنید، به‌خصوص اگر علائم دارید.\n"
        if notes:
            msg += f"\nیادداشت: {notes}\n"
        return msg

    items = data.get("items") or []
    usable = 0
    flagged_lines: List[str] = []
    normal_lines: List[str] = []
    missing_range: List[str] = []

    for it in items:
        name = str(it.get("name") or "").strip()
        value_s = str(it.get("value") or "").strip()
        unit = str(it.get("unit") or "").strip()
        rng = str(it.get("range") or "").strip()

        if not name or not value_s:
            continue

        value = parse_float(value_s)
        if rng:
            low, high = range_to_tuple(rng)
        else:
            low, high = (None, None)

        if value is None:
            line = f"• {name}: {value_s} {unit}".strip()
            normal_lines.append(line)
            usable += 1
            continue

        usable += 1

        if low is None and high is None:
            missing_range.append(f"• {name}: {value} {unit} (بدون محدوده مرجع)")
            continue

        status = classify(value, low, high)
        line = f"• {name}: {value} {unit} | محدوده: {rng} → **{status}**"
        if status == "نرمال":
            normal_lines.append(line)
        else:
            flagged_lines.append(line)

    msg = profile_line + "🧾 **بررسی اولیه نتایج (اطلاعاتی، نه تشخیص پزشکی):**\n\n"

    if usable == 0:
        msg += (
            "متأسفانه نتوانستم آیتم‌های قابل خواندن از گزارش استخراج کنم.\n"
            "✅ اگر عکس است، لطفاً PDF ارسال کنید یا متن نتایج را کپی کنید.\n"
        )
        return msg

    if flagged_lines:
        msg += "⚠️ **موارد خارج از محدوده:**\n" + "\n".join(flagged_lines[:30]) + "\n\n"
    else:
        msg += "✅ **مورد خارج از محدوده پیدا نشد (بر اساس محدوده‌های موجود در برگه).**\n\n"

    if missing_range:
        msg += "ℹ️ **مواردی که محدوده مرجع نداشتند:**\n" + "\n".join(missing_range[:30]) + "\n\n"

    if normal_lines:
        msg += "📍 **موارد دیگر/نرمال:**\n" + "\n".join(normal_lines[:20]) + "\n\n"

    msg += (
        "⚠️ **نکته مهم:** محدوده‌های مرجع بر اساس سن/جنس/آزمایشگاه متفاوت است. "
        "اگر علائم شدید دارید یا نتیجه خیلی غیرعادی است، با پزشک/اورژانس مشورت کنید."
    )
    return msg


# -------------------------
# Commands: profile
# -------------------------
def normalize_sex(s: str) -> Optional[str]:
    s = s.strip().lower()
    if s in ["male", "m", "man", "مرد", "آقا", "پسر"]:
        return "مرد"
    if s in ["female", "f", "woman", "زن", "خانم", "دختر"]:
        return "زن"
    return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام 👋\n"
        "من دستیار اطلاعات پزشکی هستم (پزشک نیستم) و فقط فارسی صحبت می‌کنم.\n\n"
        "✅ اول بهتره پروفایل‌تو تنظیم کنی:\n"
        "مثال:\n"
        "/profile 35 مرد\n"
        "/profile 28 زن\n\n"
        "سپس می‌تونی:\n"
        "• متن علائم یا سوال پزشکی بفرستی\n"
        "• PDF آزمایش/ادرار بفرستی\n"
        "• عکس واضح از نتیجه بفرستی (اگر OCR نصب باشد)\n\n"
        "⚠️ اگر وضعیت اورژانسی است (درد قفسه سینه، تنگی نفس شدید، بیهوشی)، فوراً با اورژانس تماس بگیر."
    )

async def profile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Usage: /profile 35 مرد
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "فرمت درست:\n"
            "/profile سن جنسیت\n\n"
            "مثال:\n"
            "/profile 35 مرد\n"
            "/profile 28 زن"
        )
        return

    age_str = args[0]
    sex_str = " ".join(args[1:])

    try:
        age = int(re.search(r"\d+", age_str).group(0))  # type: ignore
        if not (0 < age < 120):
            raise ValueError()
    except Exception:
        await update.message.reply_text("سن نامعتبر است. مثال: /profile 35 مرد")
        return

    sex = normalize_sex(sex_str)
    if not sex:
        await update.message.reply_text("جنسیت را درست وارد کنید: «مرد» یا «زن»\nمثال: /profile 35 مرد")
        return

    set_profile(update, age, sex)
    await update.message.reply_text(f"✅ ذخیره شد: {sex}، {age} ساله")

async def myprofile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    p = get_profile(update)
    if not p:
        await update.message.reply_text("پروفایلی ذخیره نشده. مثال:\n/profile 35 مرد")
        return
    await update.message.reply_text(f"👤 پروفایل شما:\nسن: {p.get('age')}\nجنسیت: {p.get('sex')}")

async def resetprofile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_profile(update)
    await update.message.reply_text("🗑️ پروفایل پاک شد. دوباره می‌تونی تنظیمش کنی:\n/profile 35 مرد")


# -------------------------
# Handlers: text / document / photo
# -------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.message.text or "").strip()
    p = get_profile(update)

    # quick emergency note
    if ("درد قفسه سینه" in user_text) and ("تنگی نفس" in user_text):
        await update.message.reply_text(
            "⚠️ درد قفسه سینه همراه تنگی نفس می‌تواند اورژانسی باشد.\n"
            "لطفاً همین الان با اورژانس تماس بگیرید یا به نزدیک‌ترین اورژانس مراجعه کنید.\n\n"
            "اگر می‌توانید بفرمایید:\n"
            "• از چه زمانی شروع شده؟\n"
            "• شدت ۱ تا ۱۰؟\n"
            "• تعریق/تهوع/سرگیجه دارید؟"
        )
        return

    profile_text = ""
    if p:
        profile_text = f"پروفایل کاربر: سن={p.get('age')}، جنسیت={p.get('sex')}.\n\n"

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": profile_text + f"فقط فارسی پاسخ بده.\n\nپیام کاربر: {user_text}"},
            ],
        )
        out = (resp.output_text or "").strip() or "متأسفم، مشکلی پیش آمد. لطفاً دوباره تلاش کنید."
        await update.message.reply_text(out)
    except Exception:
        logging.exception("OpenAI error")
        await update.message.reply_text("❌ خطایی رخ داد. لطفاً دوباره تلاش کنید.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return

    p = get_profile(update)
    mime = (doc.mime_type or "").lower()
    file = await doc.get_file()
    b = bytes(await file.download_as_bytearray())

    await update.message.reply_text("در حال خواندن فایل و استخراج نتایج... ⏳")

    try:
        text = ""
        if "pdf" in mime or (doc.file_name or "").lower().endswith(".pdf"):
            text = extract_text_from_pdf(b)
        elif mime.startswith("image/"):
            text = extract_text_from_image(b)

        if not text.strip():
            if mime.startswith("image/") and not OCR_AVAILABLE:
                await update.message.reply_text(
                    "من عکس را دریافت کردم، اما OCR نصب نیست و نمی‌توانم متن را بخوانم.\n\n"
                    "✅ بهترین کار: فایل PDF نتیجه را ارسال کنید.\n"
                    "یا OCR را نصب کنید (اختیاری)."
                )
            else:
                await update.message.reply_text("متأسفانه متن قابل استخراج نبود. لطفاً PDF واضح‌تری ارسال کنید.")
            return

        structured = openai_extract_structured(text, p)
        answer = build_flag_report(structured, p)
        await update.message.reply_text(answer)

    except Exception:
        logging.exception("Failed to process document")
        await update.message.reply_text("❌ پردازش فایل ناموفق بود. لطفاً PDF واضح‌تری ارسال کنید.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photos = update.message.photo
    if not photos:
        return

    p = get_profile(update)
    photo = photos[-1]
    file = await photo.get_file()
    b = bytes(await file.download_as_bytearray())

    if not OCR_AVAILABLE:
        await update.message.reply_text(
            "عکس دریافت شد ✅\n"
            "اما OCR نصب نیست و نمی‌توانم مقادیر را دقیق بخوانم.\n\n"
            "✅ لطفاً PDF نتیجه را بفرستید (بهترین گزینه)."
        )
        return

    await update.message.reply_text("در حال خواندن عکس و استخراج نتایج... ⏳")

    try:
        text = extract_text_from_image(b)
        if not text.strip():
            await update.message.reply_text("متأسفانه متن قابل استخراج نبود. لطفاً عکس واضح‌تر یا PDF ارسال کنید.")
            return

        structured = openai_extract_structured(text, p)
        answer = build_flag_report(structured, p)
        await update.message.reply_text(answer)

    except Exception:
        logging.exception("Failed to process photo")
        await update.message.reply_text("❌ پردازش عکس ناموفق بود. لطفاً PDF یا عکس واضح‌تر ارسال کنید.")


# -------------------------
# Main
# -------------------------
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("profile", profile_cmd))
    app.add_handler(CommandHandler("myprofile", myprofile_cmd))
    app.add_handler(CommandHandler("resetprofile", resetprofile_cmd))

    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logging.info("Bot started (Persian + profiles enabled)")
    app.run_polling()

if __name__ == "__main__":
    main()
