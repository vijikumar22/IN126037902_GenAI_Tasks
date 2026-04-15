from langchain_core.prompts import PromptTemplate

# Skill Extraction Prompt
extraction_prompt_template = """
You are an expert HR parser. Your task is to extract the following information from the provided resume:
1. Skills
2. Experience (in years, or relevant job titles)
3. Tools/Frameworks

Resume:
{resume_text}

Output your response in a structured JSON format with the following keys:
- "skills": list of strings (skills)
- "experience": string (description of experience)
- "tools": list of strings (tools and frameworks)

Do NOT assume skills not present in the resume. Only extract information explicitly mentioned.

JSON Output:
"""

extraction_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template=extraction_prompt_template
)

# Matching and Scoring Prompt
matching_prompt_template = """
You are an expert AI Recruiter evaluating a candidate for a job.
You are provided with the candidate's extracted profile (skills, experience, tools) and the Job Description.

Job Description:
{job_description}

Candidate Profile:
{candidate_profile}

Task:
1. Compare the candidate's profile with the job requirements.
2. Assign a fit score from 0 to 100 based on how well the candidate matches the requirements.
3. Provide a clear reasoning/explanation for why this score was assigned. Consider skill match, experience match, and any gaps.

Output your response in a structured JSON format with the following keys:
- "score": integer (0 to 100)
- "explanation": string (your reasoning)

JSON Output:
"""

matching_prompt = PromptTemplate(
    input_variables=["job_description", "candidate_profile"],
    template=matching_prompt_template
)
