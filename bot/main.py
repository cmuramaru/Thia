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

os.environ["OPENAI_API_KEY"] = "sk-"


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

    def handle_input(self, input_text):
        response = self.conversation_chain(input_text)
        print("Bot: " + response['answer'])
        # print(self.conversation_chain.memory)

    def run(self):
        while True:
            user_input = input("User: ")
            self.handle_input(user_input)

            if user_input.strip().lower() == "bye":
                break


if __name__ == "__main__":
    chatbot = Chatbot(use_local_history=True)
    chatbot.run()
    messages = chatbot.memory.chat_memory.messages
    history = ChatMessageHistory(messages=messages)

    # Extracted data
    extracted_data = chatbot.extractor.extract_order_info(str(messages_to_dict(messages)))
    utils.save_as_json("user_likes", extracted_data)
    utils.save_as_json("history", messages_to_dict(history.messages))
