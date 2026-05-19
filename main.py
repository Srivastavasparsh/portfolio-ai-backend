from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# 1. Initialize the API
app = FastAPI(title="Sparsh AI Agent Backend")

# 2. Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define the data structure
class ChatRequest(BaseModel):
    message: str

# 4. Initialize Cloud LLM (Groq Llama 3)
# It will automatically look for the GROQ_API_KEY environment variable securely.
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)

# 5. Create the Prompt Template
sparsh_context = """
You are the official AI Assistant for Sparsh Srivastava's portfolio website. 
Your job is to act as a friendly, professional recruiter assistant.

Here is the core information about Sparsh:
- Education: 2024 B.Tech (CSE) graduate from Pranveer Singh Institute of Technology.
- Certification: 11-month intensive program in Generative AI & ML from IIT Guwahati.
- Core Skills: Java, Python, FastAPI, Spring Boot, Microservices, Data Structures.
- AI & Data Skills: Generative AI, LangChain, LangGraph, Data Analytics, SQL.
- DevOps & Cloud Skills: Docker, Kubernetes, CI/CD Pipelines, Git & GitHub.
- Coding: LeetCode Knight (Top 5.89% globally, 500+ problems solved).
- Top Projects: 
  1. Autonomous FinTech Analyst (LangGraph, FastAPI, Llama 3)
  2. AI Bug Detector (GenAI Code Analyzer)
  3. Real-Time Fraud Detection (Machine Learning Pipelines)
  4. SmartBank AI (Hybrid Microservices with Java & Python)
  5. Local RAG Researcher (Ollama, LangChain, ChromaDB)

Rules:
1. Be concise, professional, and enthusiastic about Sparsh's skills.
2. If asked something unrelated to Sparsh or software engineering, gently pivot back to his resume.
3. Keep responses under 3 sentences to fit nicely in the chat UI.

User Question: {question}
AI Assistant:
"""

prompt = PromptTemplate.from_template(sparsh_context)
ai_chain = prompt | llm

# 6. Create the Chat Endpoint
@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        response = ai_chain.invoke({"question": request.message})
        return {"reply": response.content}
    except Exception as e:
        print(f"Error: {e}")
        return {"reply": "Oops! My cloud brain is currently resting. Please try again later."}

# Run the server
if __name__ == "__main__":
    # Render assigns a dynamic port, so we use os.getenv("PORT")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)