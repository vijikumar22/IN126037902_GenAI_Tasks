# ==========================================
# RAG Customer Support Assistant (FINAL)
# LangGraph + ChromaDB + Ollama + HITL
# ==========================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import StateGraph

# ==============================
# LOAD PDF
# ==============================
loader = PyPDFLoader("data.pdf")
docs = loader.load()

# ==============================
# CHUNKING
# ==============================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

# ==============================
# EMBEDDINGS + VECTOR DB
# ==============================
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ==============================
# LLM (Better model)
# ==============================
llm = ChatOllama(model="mistral") # 🔥 better than phi

# ==============================
# ANSWER FUNCTION
# ==============================
def generate_answer(query, context):
    if not context.strip():
        return "I don't know"

    prompt = f"""
You are a customer support assistant.

Answer ONLY from the context below.
Give a short and direct answer.

Context:
{context}

Question:
{query}
"""

    try:
        response = llm.invoke(prompt)

        # 🔥 ChatOllama returns .content
        answer = response.content.strip()

        if answer == "":
            return "I don't know"

        return answer

    except Exception as e:
        print("❌ LLM Error:", e)
        return "I don't know"
# ==============================
# STATE
# ==============================
class State(dict):
    pass

# ==============================
# PROCESS NODE
# ==============================
def process_node(state):
    query = state.get("query", "")

    docs = retriever.invoke(query)

    print("\n🔍 Retrieved Chunks:", len(docs))

    context = " ".join([d.page_content for d in docs])

    print("\n📄 Context Preview:", context[:200])

    answer = generate_answer(query, context)

    confidence = len(docs)

    return {
        "query": query,
        "context": context,
        "answer": answer,
        "confidence": confidence
    }

# ==============================
# ROUTER
# ==============================
def router(state):
    if state.get("answer", "") == "I don't know":
        return "hitl"
    return "output"
# ==============================
# HITL NODE
# ==============================
def hitl_node(state):
    print("\n⚠️ Escalating to Human Agent...")
    human_answer = input("👨‍💻 Enter human response: ")

    return {
        "query": state.get("query", ""),
        "context": state.get("context", ""),
        "answer": human_answer,
        "confidence": 0
    }

# ==============================
# OUTPUT NODE
# ==============================
def output_node(state):
    print("\n✅ Final Answer:")

    answer = state.get("answer")

    if not answer:
        print("❌ No answer generated")
    else:
        print("👉", answer)

    return state
# ==============================
# BUILD GRAPH
# ==============================
builder = StateGraph(State)

builder.add_node("process", process_node)
builder.add_node("hitl", hitl_node)
builder.add_node("output", output_node)

builder.set_entry_point("process")

builder.add_conditional_edges(
    "process",
    router,
    {
        "hitl": "hitl",
        "output": "output"
    }
)

builder.add_edge("hitl", "output")

graph = builder.compile()

# ==============================
# RUN LOOP
# ==============================
print("🚀 RAG Customer Support Assistant Started")

while True:
    query = input("\n💬 Ask your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    graph.invoke({"query": query})