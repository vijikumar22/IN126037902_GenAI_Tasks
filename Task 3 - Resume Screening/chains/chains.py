import json
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from prompts.prompts import extraction_prompt, matching_prompt

@traceable(name="Evaluate_Candidate_Pipeline")
def evaluate_candidate(resume_text: str, job_description: str, llm: ChatGroq):
    """
    Runs the full resume screening pipeline: Extract -> Match -> Score -> Explain
    """
    json_parser = JsonOutputParser()
    
    # Step 1: Extract Skills & Experience
    extraction_chain = extraction_prompt | llm | json_parser
    candidate_profile = extraction_chain.invoke({"resume_text": resume_text})
    
    # Step 2: Match and Score
    matching_chain = matching_prompt | llm | json_parser
    evaluation = matching_chain.invoke({
        "job_description": job_description,
        "candidate_profile": json.dumps(candidate_profile)
    })
    
    return {
        "extracted_profile": candidate_profile,
        "score": evaluation.get("score"),
        "explanation": evaluation.get("explanation")
    }
