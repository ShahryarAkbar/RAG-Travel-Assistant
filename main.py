import json
from pathlib import Path

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain_community.llms.fake import FakeListLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import initialize_agent


import os
from dotenv import load_dotenv

load_dotenv()  


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
            continue  
        if "Tokyo" in query and flight["to"] == "Tokyo":
            result.append(flight)

    if not result:
        return "No matching flights found."
    return json.dumps(result, indent=2)


def setup_visa_qa():
    loader = TextLoader("data/visa_rules.md")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    # llm = FakeListLLM(responses=["This is a simulated assistant reply."])
    # embeddings = FakeEmbeddings(size=1536)
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vectorstore = setup_visa_qa()

@tool
def visa_refund_qa(question: str) -> str:
    """Answer visa or refund-related questions."""
    docs = vectorstore.similarity_search(question, k=2)
    return "\n\n".join([doc.page_content for doc in docs])


llm = ChatOpenAI(temperature=0)

tools = [search_flights, visa_refund_qa]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)


if __name__ == "__main__":
    print("🧳 Welcome to the Conversational Travel Assistant!")
    print("Ask me about flights, visa policies, or refund rules.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.run(user_input)
        print(f"\nAssistant: {response}\n")
