import os
import json
import sqlite3
import requests
import pandas as pd

from flask import Flask, request
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = Flask(__name__)
client = OpenAI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

with open("handoff_numbers.json", "r", encoding="utf-8") as f:
    HANDOFF_NUMBERS = json.load(f)

with open("knowledge.txt", "r", encoding="utf-8") as f:
    BUSINESS_RULES = f.read()

DATASET_PATH = "jal_yoga_qa_dataset_sg_500_each.csv"
df = pd.read_csv(DATASET_PATH)
df["user_message"] = df["user_message"].fillna("").astype(str)
df["ideal_bot_response"] = df["ideal_bot_response"].fillna("").astype(str)
df["segment"] = df["segment"].fillna("").astype(str)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
message_matrix = vectorizer.fit_transform(df["user_message"])

WELCOME_MESSAGE = """Namaste! Thank you for reaching out to Jal Yoga. 🙏

Please let us know what you're looking for today:
1. Schedule a Trial
2. I’m a current member
3. I’d like to find out more about Jal Yoga
4. Corporate/Partnerships
5. Staff Hub"""

DB_FILE = "chatbot.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            phone TEXT PRIMARY KEY,
            history TEXT NOT NULL,
            awaiting_handoff_area INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def get_session(phone):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT history, awaiting_handoff_area FROM sessions WHERE phone = ?",
        (phone,)
    )
    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "history": json.loads(row[0]),
            "awaiting_handoff_area": bool(row[1])
        }

    return {
        "history": [],
        "awaiting_handoff_area": False
    }


def save_session(phone, session):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO sessions (phone, history, awaiting_handoff_area)
        VALUES (?, ?, ?)
        ON CONFLICT(phone) DO UPDATE SET
            history = excluded.history,
            awaiting_handoff_area = excluded.awaiting_handoff_area
    """, (
        phone,
        json.dumps(session["history"][-12:], ensure_ascii=False),
        1 if session["awaiting_handoff_area"] else 0
    ))
    conn.commit()
    conn.close()


def retrieve_examples(user_text, top_k=3):
    query_vec = vectorizer.transform([user_text])
    scores = cosine_similarity(query_vec, message_matrix).flatten()
    best_idx = scores.argsort()[-top_k:][::-1]

    examples = []
    for i in best_idx:
        if scores[i] <= 0:
            continue
        row = df.iloc[i]
        examples.append({
            "segment": row["segment"],
            "user_message": row["user_message"],
            "ideal_bot_response": row["ideal_bot_response"]
        })
    return examples


def history_to_text(history):
    lines = []
    for item in history[-8:]:
        lines.append(f'{item["role"].upper()}: {item["content"]}')
    return "\n".join(lines)


def build_examples_text(examples):
    if not examples:
        return "No close dataset examples found."
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"Example {i}\n"
            f"Segment: {ex['segment']}\n"
            f"User: {ex['user_message']}\n"
            f"Assistant: {ex['ideal_bot_response']}"
        )
    return "\n\n".join(parts)


def ask_llm(user_text, session):
    examples = retrieve_examples(user_text, top_k=3)

    prompt = f"""
You are Jal Yoga Singapore's WhatsApp assistant.

Business rules:
{BUSINESS_RULES}

Closest dataset examples:
{build_examples_text(examples)}

Conversation history:
{history_to_text(session["history"])}

Current user message:
{user_text}

Instructions:
- Reply in clear, short WhatsApp-friendly English.
- Ask one question at a time if information is missing.
- Use the business rules first.
- Use the dataset examples as guidance for style and typical answers.
- Do not invent phone numbers, trainer names, live slot counts, or policies.
- If the user requests a human, or issue needs manual review, refund, payment check, complaint handling, account-specific access, or you are unsure, start your reply with exactly __HANDOFF__ followed by a space and the message.
- Otherwise, return only the reply text.
""".strip()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )

    return response.output_text.strip()


def area_from_text(text):
    text = text.lower().strip()
    for area in HANDOFF_NUMBERS.keys():
        if area in text:
            return area
    return None


def make_wa_link(number):
    return f"https://wa.me/{number}"


def send_whatsapp_text(to_number, message_text):
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {
            "body": message_text
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print("WhatsApp response:", response.status_code, response.text)
    return response


def extract_incoming_message(payload):
    try:
        entry = payload["entry"][0]
        change = entry["changes"][0]
        value = change["value"]

        messages = value.get("messages", [])
        if not messages:
            return None, None

        message = messages[0]
        from_number = message["from"]
        msg_type = message.get("type")

        if msg_type == "text":
            text = message["text"]["body"].strip()
            return from_number, text

        if msg_type == "interactive":
            interactive = message["interactive"]
            text = (
                interactive.get("button_reply", {}).get("title")
                or interactive.get("list_reply", {}).get("title")
                or ""
            ).strip()
            return from_number, text

        return from_number, ""
    except Exception as e:
        print("extract error:", e)
        return None, None


def ask_for_handoff_area(phone, session, custom_message=None):
    session["awaiting_handoff_area"] = True
    save_session(phone, session)

    if custom_message:
        msg = custom_message + "\n\nPlease reply with: North, South, East, West, or Centre."
    else:
        msg = "I’ll connect you to our customer service team. Please reply with: North, South, East, West, or Centre."

    send_whatsapp_text(phone, msg)


def handle_handoff_area_choice(phone, user_text, session):
    area = area_from_text(user_text)
    if not area:
        send_whatsapp_text(
            phone,
            "Please reply with one of these areas: North, South, East, West, or Centre."
        )
        return

    number = HANDOFF_NUMBERS[area]
    link = make_wa_link(number)

    session["awaiting_handoff_area"] = False
    session["history"].append({"role": "user", "content": user_text})
    session["history"].append({
        "role": "assistant",
        "content": f"Customer service handoff for {area}"
    })
    save_session(phone, session)

    send_whatsapp_text(
        phone,
        f"Here is our {area.title()} customer service WhatsApp number:\n+{number}\n{link}"
    )


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200

    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def receive_webhook():
    payload = request.get_json(silent=True) or {}
    print("Incoming payload:", json.dumps(payload, indent=2))

    phone, user_text = extract_incoming_message(payload)
    if not phone:
        return "no message", 200

    session = get_session(phone)

    if session["awaiting_handoff_area"]:
        handle_handoff_area_choice(phone, user_text, session)
        return "ok", 200

    cleaned = (user_text or "").strip().lower()

    if cleaned in {"hi", "hello", "hey", "menu", "start"} and not session["history"]:
        session["history"].append({"role": "user", "content": user_text})
        session["history"].append({"role": "assistant", "content": WELCOME_MESSAGE})
        save_session(phone, session)
        send_whatsapp_text(phone, WELCOME_MESSAGE)
        return "ok", 200

    human_keywords = [
        "human", "agent", "customer service", "real person", "staff",
        "refund", "complaint", "complain", "payment issue", "billing issue"
    ]

    if any(k in cleaned for k in human_keywords):
        ask_for_handoff_area(
            phone,
            session,
            "This looks like something our customer service team should handle directly."
        )
        return "ok", 200

    session["history"].append({"role": "user", "content": user_text})

    try:
        ai_reply = ask_llm(user_text, session)
    except Exception as e:
        print("OpenAI error:", e)
        ask_for_handoff_area(
            phone,
            session,
            "I’m having trouble answering that right now."
        )
        return "ok", 200

    if ai_reply.startswith("__HANDOFF__"):
        handoff_msg = ai_reply.replace("__HANDOFF__", "", 1).strip()
        ask_for_handoff_area(phone, session, handoff_msg)
        return "ok", 200

    session["history"].append({"role": "assistant", "content": ai_reply})
    save_session(phone, session)
    send_whatsapp_text(phone, ai_reply)

    return "ok", 200


if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)