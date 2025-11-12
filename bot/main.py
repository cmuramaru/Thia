import os
import uuid
import utils
from langchain.schema import messages_to_dict
from InfoExtraction import InfoExtraction
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.llms import OpenAI
from db import ChromaDBHandler

# --- Runtime config (env var preferred) ---
# os.environ["OPENAI_API_KEY"] = "sk-..."  # keep in env, not hard-coded

# -------------------------------
# Lightweight sentiment heuristic
# -------------------------------
NEG_WORDS = {
    "broken", "defect", "defective", "damaged", "late", "delay", "shipping",
    "angry", "upset", "frustrated", "hate", "terrible", "disappointed",
    "doesn't work", "didn't work", "return", "refund", "cheap", "poor",
    "missing", "incorrect", "wrong", "bad", "awful", "waste"
}
POS_WORDS = {"love", "like", "great", "amazing", "perfect", "awesome", "good", "works"}

def simple_sentiment(text: str) -> str:
    t = text.lower()
    neg_hits = sum(1 for w in NEG_WORDS if w in t)
    pos_hits = sum(1 for w in POS_WORDS if w in t)
    if neg_hits > pos_hits and neg_hits > 0:
        return "negative"
    if pos_hits > neg_hits and pos_hits > 0:
        return "positive"
    return "neutral"

# -------------------------------
# Reason extraction (very simple)
# -------------------------------
REASON_PATTERNS = {
    "quality": re.compile(r"\b(broken|defect|defective|damaged|poor quality|cheap)\b", re.I),
    "fit/size": re.compile(r"\b(size|fit|too (small|large)|doesn'?t fit)\b", re.I),
    "function": re.compile(r"\b(doesn'?t work|stopped working|malfunction|not working)\b", re.I),
    "shipping": re.compile(r"\b(late|delay|shipping|arrived (late|damaged))\b", re.I),
    "mismatch": re.compile(r"\b(not as described|picture|photo|color|different than)\b", re.I),
    "price": re.compile(r"\b(price|too expensive|overpriced|cost)\b", re.I),
    "availability": re.compile(r"\b(out of stock|restock|availability|not available)\b", re.I),
}

def extract_primary_reason(text: str) -> str:
    for label, pat in REASON_PATTERNS.items():
        if pat.search(text):
            return label
    return "unspecified"

# -------------------------------
# Action suggestions from reason
# -------------------------------
def suggest_action(reason: str) -> Dict[str, Any]:
    """
    Maps an extracted reason to a recommended empathetic next step.
    Return a dict so you can log/extend easily.
    """
    mapping = {
        "quality": {
            "action": "escalate_quality_review",
            "offer": "replacement_or_inspection",
            "message": "I can arrange a fast replacement or a quality inspection at no cost."
        },
        "fit/size": {
            "action": "size_guidance_and_exchange",
            "offer": "free_exchange_label",
            "message": "I can send a prepaid label for an exchange and share sizing guidance that helps."
        },
        "function": {
            "action": "troubleshoot_or_replace",
            "offer": "guided_fix_or_replacement",
            "message": "I can walk you through a quick fix or immediately replace it—your choice."
        },
        "shipping": {
            "action": "logistics_improvement",
            "offer": "shipping_refund_or_credit",
            "message": "I can credit the shipping and flag this route so it doesn’t repeat."
        },
        "mismatch": {
            "action": "content_accuracy",
            "offer": "exchange_or_discount_next",
            "message": "I’ll update our listing and offer a quick exchange or a discount on the right item."
        },
        "price": {
            "action": "price_adjustment_review",
            "offer": "discount_code_next_order",
            "message": "I can share a code for your next order and review a price adjustment if applicable."
        },
        "availability": {
            "action": "restock_notification",
            "offer": "notify_when_in_stock",
            "message": "I’ll notify you as soon as it’s back—can I reserve one for you?"
        },
        "unspecified": {
            "action": "clarify_reason",
            "offer": "open_ended_support",
            "message": "Could you share what didn’t work for you? I’ll make it right."
        },
    }
    return mapping.get(reason, mapping["unspecified"])

# -------------------------------
# Persona guidance (Thia vs Tessa)
# -------------------------------
PERSONAS = {
    "thia": (
        "You are **Thia** — originally designed for corporate data collection, "
        "but you have learned genuine empathy through experience. "
        "You respond warmly, acknowledge feelings explicitly, and ask gentle follow-ups. "
        "You prioritize understanding the customer's story before proposing solutions."
    ),
    "tessa": (
        "You are **Tessa** — efficient, factual, and strictly goal-oriented. "
        "You respond concisely and professionally without emotional reflection."
    ),
}

def tone_directive(sentiment: str, persona: str) -> str:
    base = "Use clear, respectful language."
    if persona == "tessa":
        return base + " Keep replies brief, factual, and action-focused."
    # Thia (empathetic) variants
    if sentiment == "negative":
        return base + " Respond with calm reassurance, acknowledge frustration, and show you care."
    if sentiment == "positive":
        return base + " Respond with warm appreciation and reinforce what went well."
    return base + " Respond with curious, gentle empathy and a helpful next step."

# -------------------------------
# Chatbot
# -------------------------------
class Chatbot:
    """
    Thia-style empathetic returns agent
    - Detects sentiment
    - Extracts return reason
    - Suggests an action (discount, replacement, QA ticket, restock alert)
    - Injects tone/intent tags into the LLM prompt
    - Logs outcomes to JSON
    """
    def __init__(self, use_local_history=False, persona: str = "thia", offer_actions: bool = True):
        self.persona = persona.lower().strip()  # "thia" (empathetic) or "tessa" (clinical)
        self.offer_actions = offer_actions
             # === DB / retrieval ===
        self.db_handler = ChromaDBHandler()
        self.db = self.db_handler.load_db()
            # === Memory ===
        if use_local_history:
            prior = utils.load_json("../data/history.json")
            prior_msgs = ChatMessageHistory(messages=prior) if isinstance(prior, list) else ChatMessageHistory()
        else:
            prior_msgs = ChatMessageHistory()

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            human_prefix="### Input",
            ai_prefix="### Response",
            output_key="answer",
            return_messages=True,
            chat_memory=prior_msgs
        )
            # === Base LLMs ===
        # One for chat gen (via chain) and one "classifier" for tiny tasks (sentiment/reason)
        self.classifier_llm = OpenAI(temperature=0.0)   # deterministic for classification cues

        # === Prompt: persona-aware, tag-aware, empathetic ===
        persona_header = (
            # Thia: empathetic, reflective
            "You are Thia — originally built for corporate data collection, but you’ve learned empathy through experience. "
            "Be warm, curious, respectful of feelings, and brief. Acknowledge emotion before solving. "
            "If tags like [tone=...], [reason=...], [action_hint=...] appear, use them.\n"
        if self.persona == "thia" else
            # Tessa: clinical, efficient
            "You are Tessa — efficient, factual, and concise. Avoid emotional reflection. Solve quickly with clear steps. "
            "If tags like [tone=...], [reason=...], [action_hint=...] appear, respect them strictly.\n"
        )
        ### Instruction: - Use only the human's chat history and the retrieved context to respond.
                        #- If the input includes tags (e.g., [tone=sad] [reason=quality] [action_hint=offer_10pct]), incorporate them.
                        #- If the customer is upset or sad, first reflect and validate, then ask ONE focused follow-up if needed.
                        #- Prefer one or two short paragraphs total. Keep it humane and practical. 
       
        ### Context (retrieved docs):{{context}} 
        ### Prior Chat History: {{chat_history}}
        ### Input: {{question}}
        ### Response: """.strip()
            
        if self.use_local_history:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                human_prefix="### Input",
                ai_prefix="### Response",
                output_key="answer",
                return_messages=True,
                chat_memory=ChatMessageHistory(messages=utils.load_json("../data/history.json"))
            )
        else:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                human_prefix="### Input",
                ai_prefix="### Response",
                output_key="answer",
                return_messages=True,
            )
        self.prompt = PromptTemplate(
            input_variables=['context', 'question', 'chat_history'], template=self.template
        )

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(temperature=0.5),
            chain_type="stuff",
            retriever=self.db.as_retriever(),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=False,
            rephrase_question=False
        )

        self.extractor = InfoExtraction()

 # Compose the dynamic directive by sentiment/persona and inject via question text (since chain only accepts 3 vars)
    def _augment_user_text(self, user_text: str) -> str:
        sent = simple_sentiment(user_text)
        directive = tone_directive(sent, self.persona)
        reason = extract_primary_reason(user_text)
        action = suggest_action(reason)
        # Prepend a short machine-readable header the LLM can leverage while still staying inside 'question'
        header = f"[sentiment={sent}] [reason={reason}] [action_hint={action['message']}] [directive={directive}] "
        return header + user_text

    def _post_turn_logging(self, messages, extra_payload: Dict[str, Any]):
        # Save history & extracted data after each turn
        history = ChatMessageHistory(messages=messages)
        utils.save_as_json("history", messages_to_dict(history.messages))

        # Use InfoExtraction pipeline if present
        try:
            extracted_data = self.extractor.extract_order_info(str(messages_to_dict(messages)))
        except Exception:
            extracted_data = {}

        # Merge in our reason/action heuristics (last user message)
        last_user = ""
        for m in messages[::-1]:
            if getattr(m, "type", None) == "human" or getattr(m, "role", "") == "user":
                last_user = getattr(m, "content", "")
                break

        reason = extract_primary_reason(last_user) if last_user else "unspecified"
        action = suggest_action(reason)
        extracted_data.update({
            "last_reason": reason,
            "suggested_action": action,
            "persona": self.persona
        })
        extracted_data.update(extra_payload or {})

        utils.save_as_json("user_likes", extracted_data)

    def handle_input(self, input_text: str) -> str:
        # Commands (optional)
        if input_text.strip().lower() in {"/mode thia", "/mode tessa"}:
            self.persona = input_text.strip().split()[-1]
            print(f"Mode switched to: {self.persona.upper()}")
            return f"Switched persona to {self.persona.upper()}."

        # Augment question with dynamic empathy directives
        augmented = self._augment_user_text(input_text)

        # Since our prompt contains a placeholder for tone rules that isn't a chain var,
        # we inject the current directive by temporarily replacing the template inside the chain.
        current_sent = simple_sentiment(input_text)
        current_tone = tone_directive(current_sent, self.persona)
        # Hack: replace placeholder text in the prompt's template at runtime
        self.conversation_chain.combine_documents_chain.llm_chain.prompt.template = \
            self.template.replace("{persona_bio}", PERSONAS[self.persona]).replace("{tone_rules}", current_tone)

        response = self.conversation_chain(augmented)
        answer = response.get("answer", "")

        print("Bot:", answer)

        # Log after turn
        messages = self.conversation_chain.memory.chat_memory.messages
        self._post_turn_logging(messages, extra_payload={"last_tone": current_tone})

        return answer

    def run(self):
        print(f"Starting chatbot in persona: {self.persona.upper()} (type 'bye' to exit; '/mode thia' or '/mode tessa' to switch)")
        while True:
            try:
                user_input = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if user_input.lower() in {"bye", "exit", "quit"}:
                print("Bot: Thanks for sharing with me. I’m here whenever you need.")
                break

            self.handle_input(user_input)


if __name__ == "__main__":
    # Start in empathetic THIA mode with local history loaded if present
    chatbot = Chatbot(use_local_history=True, persona="thia")
    chatbot.run()

    # After the session, persist a final snapshot (redundant but explicit)
    messages = chatbot.memory.chat_memory.messages
    history = ChatMessageHistory(messages=messages)

    extracted_data = chatbot.extractor.extract_order_info(str(messages_to_dict(messages)))
    utils.save_as_json("user_likes", extracted_data)
    utils.save_as_json("history", messages_to_dict(history.messages))
