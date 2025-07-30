import json
from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import initialize_agent

load_dotenv()


FLIGHT_DATA_PATH = Path("data/flights.json")

def load_flights():
    with open(FLIGHT_DATA_PATH, "r") as f:
        return json.load(f)


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", system="You are a helpful travel assistant.")


def parse_flight_query(query: str) -> dict:
    """Use the LLM to extract flight filters from a user query."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract structured flight search filters from the user's query. "
                   "Return a JSON with keys like: from, to, departure_date, return_date, alliance, avoid_overnight (true/false), refundable (true/false)."),
        ("human", "{query}")
    ])
    messages = prompt.format_messages(query=query)
    response = llm.invoke(messages)

    try:
        return json.loads(response.content)
    except:
        return {}


@tool
def search_flights(query: str) -> str:
    """Search flights based on a natural language query."""
    flights = load_flights()
    filters = parse_flight_query(query)
    result = []

    for flight in flights:
        if filters.get("to") and filters["to"].lower() != flight["to"].lower():
            continue
        if filters.get("from") and filters["from"].lower() != flight["from"].lower():
            continue
        if filters.get("alliance") and filters["alliance"].lower() != flight["alliance"].lower():
            continue
        if filters.get("avoid_overnight") and len(flight["layovers"]) > 0:
            continue
        if filters.get("refundable") is True and not flight.get("refundable", False):
            continue
        
        result.append(flight)

    if not result:
        return "No matching flights found."

    formatted = []
    for f in result:
        formatted.append(
            f" {f['airline']} ({f['alliance']})\n"
            f"From: {f['from']} → {f['to']}\n"
            f"Departure: {f['departure_date']} | Return: {f['return_date']}\n"
            f"{'Layovers: ' + ', '.join(f['layovers']) if f['layovers'] else 'Direct flight'}\n"
            f"Price: ${f['price_usd']} | {' Refundable' if f['refundable'] else ' Non-refundable'}\n"
        )
    return "\n---\n".join(formatted)


def setup_visa_qa():
    loader = TextLoader("data/visa_rules.md")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vectorstore = setup_visa_qa()

@tool
def visa_refund_qa(question: str) -> str:
    """Answer visa or refund-related questions."""
    docs = vectorstore.similarity_search(question, k=2)
    return "\n\n".join([f"{doc.page_content.strip()}" for doc in docs])


tools = [search_flights, visa_refund_qa]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)


if __name__ == "__main__":
    print("Welcome to the Conversational Travel Assistant!")
    print("Ask me about flights, visa policies, or refund rules.\n(Type 'exit' or 'quit' to leave)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Safe travels.")
            break
        response = agent.run(user_input)
        print(f"\nAssistant: {response}\n")
