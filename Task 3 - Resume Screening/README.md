# 🚀 AI Resume Screening System 

## 📌 Project Overview
This project is an AI-powered Resume Screening System built as part of the **GenAI & LLM Engineering Task**. The pipeline automatically parses a list of candidate resumes, evaluates them against a given job description, and outputs a candidate fit score along with a detailed explanation reasoning out the evaluation.

## 🛠️ Technology Stack
- **Python**: Core programming language.
- **LangChain**: Used for creating the prompt templating and tying together the evaluation pipeline utilizing Langchain Expression Language (LCEL).
- **Groq**: Using the lightning-fast open-source LLM `llama-3.3-70b-versatile` to process and evaluate resumes deterministically.
- **LangSmith**: Embedded for deep tracing, application observability, and pipeline debugging.

## ⚙️ Architecture Pipeline
The architecture follows a modular LLM pipeline:
> **Resume** → **Skill Extraction** (via JSON Output Parsers) → **Matching Logic** → **Scoring (0-100)** → **Explanation generation** 

## 📂 Project Structure
```text
Task 3 - Resume Screening/
│
├── data/
│   ├── jd.txt                  # Job Description Requirements
│   ├── resume_strong.txt       # Example: Highly-qualified candidate 
│   ├── resume_average.txt      # Example: Partially-qualified candidate
│   └── resume_weak.txt         # Example: Unqualified candidate
│
├── chains/
│   └── chains.py               # Contains the LLM Extraction & Matching logic
│
├── prompts/
│   └── prompts.py              # Zero-shot prompts for strict Extraction & Recruiter evaluation
│
├── .env                        # Private API Key configurations
├── requirements.txt            # Python dependencies
└── main.py                     # Primary pipeline execution script
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Ensure you create a `.env` file containing your active keys before executing:
```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=resume_screening_assignment
```

### 3. Run Pipeline
```bash
python main.py
```
This script will parse the `data/` folder, run evaluations on 3 different candidate resumes using Groq, and print structured JSON outputs including Score and Explanation logs. Traces are simultaneously intercepted and mapped inside LangSmith.

## 🔍 Tracing & Explainability
This project explicitly fulfills the explainability constraints. All extraction, matching, and scoring tasks are fully traceable down to the chain level utilizing LangSmith decorators (`@traceable`).
