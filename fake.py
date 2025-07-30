import json
from pathlib import Path
from typing import Any, List

from langchain_core.tools import tool
from langchain_core.language_models import LanguageModel
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import initialize_agent
from langchain_core.documents import Document

# --- 1. Load Flights Data ---
FLIGHT_DATA_PATH = Path("data/flights.json")

def load_flights():
    with open(FLIGHT_DATA_PATH, "r") as f:
        return json.load(f)

@tool
def search_flights(query: str) -> str:
    """Search flights based on a natural language query."""
    flights = load_flights()
    result = []

    for flight in flights:
        if (
            "Tokyo" in query and flight["to"] == "Tokyo" and
            "Star Alliance" in query and flight["alliance"] == "Star Alliance" and
            "overnight" in query and len(flight["layovers"]) == 0
        ):
            continue  # Skip overnight layovers if user wants to avoid
        if "Tokyo" in query and flight["to"] == "Tokyo":
            result.append(flight)

    if not result:
        return "No matching flights found."
    return json.dumps(result, indent=2)

# --- 2. Fake Embeddings ---
class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[1.0] * 10 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [1.0] * 10

# --- 3. Load and Embed Visa Data ---
def setup_visa_qa():
    loader = TextLoader("data/visa_rules.md")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = FakeEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vectorstore = setup_visa_qa()

@tool
def visa_refund_qa(question: str) -> str:
    """Answer visa or refund-related questions."""
    docs = vectorstore.similarity_search(question, k=2)
    return "\n\n".join([doc.page_content for doc in docs])

# --- 4. Fake LLM ---
class FakeLLM(LanguageModel):
    def __init__(self, response: str = "This is a fake LLM response."):
        self.response = response

    def predict(self, text: str, **kwargs: Any) -> str:
        return self.response

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        return self.response

    @property
    def _llm_type(self) -> str:
        return "fake-llm"

# --- 5. Run the Agent ---
if __name__ == "__main__":
    print("🧳 Welcome to the Conversational Travel Assistant (Fake Mode)")
    print("Ask me about flights or visa policies. Type 'exit' to quit.\n")

    llm = FakeLLM(response="(This is a simulated response from the assistant.)")
    tools = [search_flights, visa_refund_qa]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.run(user_input)
        print(f"\nAssistant: {response}\n")
