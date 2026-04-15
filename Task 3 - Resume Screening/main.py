import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from chains.chains import evaluate_candidate

def load_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Ensure GROQ_API_KEY and LangSmith vars are set
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY is not set in the environment or .env file.")
        print("Please check your .env file.")
        return

    # Initialize the LLM
    # using temperature=0 to ensure consistent, deterministic outputs for scoring
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # Load input data
    job_description = load_file("data/jd.txt")
    
    resumes = {
        "Strong Candidate": "data/resume_strong.txt",
        "Average Candidate": "data/resume_average.txt",
        "Weak Candidate": "data/resume_weak.txt"
    }
    
    print("======================================================")
    print(" AI Resume Screening System with LangSmith Tracing ")
    print("======================================================\n")

    for candidate_type, filepath in resumes.items():
        print(f"--- Evaluating {candidate_type} ---")
        resume_text = load_file(filepath)
        
        try:
            # Run the pipeline (this invocation will be traced in LangSmith)
            result = evaluate_candidate(
                resume_text=resume_text, 
                job_description=job_description,
                llm=llm
            )
            
            print(f" Candidate   : {candidate_type}")
            print(f" Score       : {result['score']}/100")
            print(f" Explanation : {result['explanation']}\n")
            
        except Exception as e:
            print(f" Error processing {candidate_type}: {e}\n")

    print(" All candidates processed.")
    print(" Check your LangSmith dashboard to view the traces for these requests!")

if __name__ == "__main__":
    main()
