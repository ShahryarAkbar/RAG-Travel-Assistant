
#  Conversational Travel Assistant 

This project is a conversational travel assistant designed to simulate an intelligent chatbot that helps users plan international travel. It supports natural language interaction for:

- Flight search and filtering based on preferences
- Visa and refund policy question answering via Retrieval-Augmented Generation (RAG)

Built using **LangChain**, **OpenAI**, and **FAISS**.

---

##  Project Structure

```
travel-assistant/
│
├── main.py               # Main chatbot implementation
├── requirements.txt      # Required Python packages
├── README.md             # Project overview and usage
│
└── data/
    ├── flights.json      # Sample flight data (mock)
    └── visa_rules.md     # Visa and refund policy documents
```

---

##  Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/username/travel-assistant.git
cd travel-assistant
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set OpenAI API Key**

Create a `.env` file or set key in the terminal:

```bash
# In .env file (recommended)
OPENAI_API_KEY=openai_key

# Or set in terminal
export OPENAI_API_KEY=openai_key       # macOS/Linux
set OPENAI_API_KEY=openai_key          # Windows
```

4. **Run the assistant**

```bash
python main.py
```

---

##  How It Works

This assistant uses a LangChain agent with two tools:

1. **Flight Search Tool**
   - Parses natural language queries to filter mock flight data (`flights.json`)

2. **Visa/Refund QA Tool**
   - Uses FAISS vector store to retrieve answers from `visa_rules.md`

---

##  Sample Queries

```text
You: Find me a round trip to Tokyo in August via Star Alliance. Avoid overnight layovers.
Assistant: Turkish Airlines – Dubai to Tokyo | Dates: Aug 15–30 | $950 | Layover: Istanbul

You: Do UAE citizens need a visa to travel to Japan?
Assistant: UAE passport holders can enter Japan visa-free for up to 30 days.
```
---
Thanks for reviewing this project!
