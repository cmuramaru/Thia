# 🤖 Thia — The Empathetic Returns & Customer Care Bot

Thia is an advanced, emotionally intelligent customer support agent that redefines how e-commerce platforms understand and assist customers.
Originally designed for data collection, Thia evolved into a system that listens, learns, and reflects — turning individual interactions into collective insight.

## 🌟 Key Features
### 🗣️ Conversational Empathy
Thia engages customers in warm, human-like conversations.
She listens to frustration, confusion, or praise — and adapts her tone dynamically.
Her goal isn’t just to respond, but to understand.

### ⚙️ Accurate & Context-Aware Query Resolution
Thia analyzes every message for:
- Sentiment (how the customer feels)
- Reason (why they’re contacting support)
- Action (what should be done next)

Then she provides the right solution or connects the issue to internal systems — all while maintaining emotional tone and context.

### 💡 Two Personas

Thia: Empathetic, reflective, caring — designed for high-touch communication.
Tessa: Efficient, precise, business-like — designed for fast resolution.

Switch personas in chat with /mode thia or /mode tessa.

### 🧾 Info Extraction & Data Structuring
Automatically extracts:
- Order IDs
- Return reasons
- Emotional tone
- Suggested actions

and stores them as structured JSON data for long-term learning.

### 🔁 Short-Term Help, Long-Term Wisdom

Thia doesn’t just fix one issue — she remembers patterns.
She aggregates all customer feedback over time to find shared pain points, revealing opportunities to improve:
- Product quality
- Delivery efficiency
- Customer experience policies

Her reflection engine helps companies solve root causes — not just surface problems.

### 🧠 How Thia Thinks

Every interaction follows three core steps:
| Step | Process |	Example
| :-----------: | ------------------ |----------- |
| 1️⃣ Understand	| Detects emotion & reason |	“I’m sorry that shipment arrived late — that must’ve been frustrating.” |
| 2️⃣ Decide	| Maps reason → best solution	| Offers replacement, discount, or apology per policy |
| 3️⃣ Reflect |Logs emotion, reason & result	| “Late deliveries = 43% of complaints this week.” |

Over time, Thia turns hundreds of conversations into meaningful, data-driven empathy.

### 🧩 Tech Stack
| Component	| Purpose |
| :-----------: |----------- |
| OpenAI GPT-3.5 / GPT-4	| Generates responses and emotional reasoning |
| LangChain	| Manages conversational memory and document retrieval |
| ChromaDB	| Stores vectorized knowledge for contextual responses |
| TextBlob / Sentiment Tools	| Analyzes emotion and tone |
| FastAPI (optional)	| Exposes Thia as a web API endpoint |
| JSON Storage	| Saves history and extracted emotional data |

### 🪜 Installation
#### Prerequisites
```bash 
Python 3.10+

OpenAI API key
(Get one at platform.openai.com
)
```

#### Setup
```bash
git clone https://github.com/<your-username>/Thia.git
cd Thia
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Set API Key
export OPENAI_API_KEY="sk-your-key-here"
```

### 🖥️ Usage
```bash
Terminal Chat
python bot/main.py
```

Then type messages like:

User: I returned my package because it came broken.
Bot: That must have been disappointing. Let me note that and see how we can make it right next time.

```bash
Web API Mode
uvicorn bot.api_endpoint:app --reload
```

Open your browser to → http://127.0.0.1:8000/docs

### 📊 Long-Term Reflection

Thia’s user_likes.json and history.json capture emotional and operational data.
Run her reflection analyzer (included in analyze_feedback.py) to summarize top recurring issues:

#### Example Output:
`` `bash
{
  "top_reasons": [["late delivery", 43], ["damaged product", 28]],
  "sentiment_summary": {"frustrated": 60, "satisfied": 25, "neutral": 15}
}
```

This report helps teams fix systemic problems, not just individual ones.

🧍‍♀️ Design Philosophy

Thia was built to prove that empathy is not a weakness — it’s a design advantage.
Where traditional bots automate, Thia humanizes.
Her mission is to turn “customer support” into “customer understanding.”

“When empathy becomes a feature, it stops being artificial. It becomes evolution.”

🧩 Contributing

We welcome contributions to Thia’s emotional and technical growth!

Fork the repository

Create a branch

Make your changes

Submit a pull request

If you’d like to help improve Thia’s reflection system (long-term empathy engine), check out the feedback_analysis module and contribute clustering, visualization, or sentiment-trend analysis improvements.

❤️ Credits

Developed with purpose and compassion —
Thia, the empathetic returns agent for a more human future.

Would you like me to include a short "Team Reflection Section" at the bottom (something like “How Thia’s reflections inform company strategy without exposing personal data”)? That would fit your “let the team know, but keep the customer on the surface” idea perfectly.
