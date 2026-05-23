"""
agents.py
---------
All the agent definitions for the Multi-Agent Orchestration System.

Three layers of agents:

Layer 1 — Department Agents (instruction routing):
    - sales_agent:       Handles sales / revenue / pricing / deals inquiries
    - hr_agent:          Handles HR / people / policy / benefits inquiries
    - operations_agent:  Handles ops / logistics / inventory / process inquiries

Layer 2 — Recruitment Analyst Agents (resume pipeline):
    - resume_analyzer:   Extracts skills, experience, education from resume text
    - skills_matcher:    Compares resume skills against JD requirements
    - question_generator: Generates interview questions based on JD + match analysis

Layer 3 — Intelligence Layer:
    - vector_agent:      Adds semantic understanding; enriches context with
                          embeddings-based similarity logic between agents

Every agent:
    1. Reads shared memory (what other agents already found)
    2. Does its job (LLM call with domain-specific prompt)
    3. Writes results back to shared memory
    4. Returns its output

This is how agents "communicate" — through the shared campfire (memory).
"""

import os
import json
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

from shared_memory import memory

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================================
# LLM HELPERS
# ============================================================================
def _call_llm_json(system: str, user: str) -> dict:
    """Call OpenAI in JSON mode. Returns parsed dict."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY missing. Add it to .env.")
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _call_llm_text(system: str, user: str) -> str:
    """Call OpenAI in normal text mode. Returns string."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY missing. Add it to .env.")
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _inject_shared_context() -> str:
    """Build a context block from shared memory for agent prompts."""
    ctx = memory.get_context_summary()
    if ctx == "No shared context available yet.":
        return ""
    return f"\n\n--- SHARED CONTEXT FROM OTHER AGENTS ---\n{ctx}\n--- END SHARED CONTEXT ---\n"


# ============================================================================
# LAYER 1: DEPARTMENT AGENTS
# ============================================================================
def sales_agent(instruction: str) -> Dict[str, Any]:
    """
    Sales Department Agent.
    Handles: pricing, deals, revenue, pipeline, quotas, proposals, client queries.
    """
    shared_ctx = _inject_shared_context()
    system = (
        "You are the SALES department AI agent. You handle inquiries about:\n"
        "- Pricing, quotes, and proposals\n"
        "- Sales pipeline and deal status\n"
        "- Revenue targets and quotas\n"
        "- Client relationship management\n"
        "- Product demos and trial requests\n"
        "- Competitive positioning\n"
        f"{shared_ctx}\n"
        "Output JSON: {'response': <your answer>, 'action_items': [list of next steps], "
        "'department': 'Sales', 'confidence': 0.0-1.0}"
    )
    result = _call_llm_json(system, instruction)
    result.setdefault("department", "Sales")
    result.setdefault("confidence", 0.8)

    # Store findings in shared memory
    memory.store("sales_agent", "last_response", result.get("response", ""))
    memory.store("sales_agent", "action_items", result.get("action_items", []))

    return result


def hr_agent(instruction: str) -> Dict[str, Any]:
    """
    HR Department Agent.
    Handles: hiring, policies, benefits, onboarding, compliance, employee relations.
    """
    shared_ctx = _inject_shared_context()
    system = (
        "You are the HR department AI agent. You handle inquiries about:\n"
        "- Hiring and recruitment processes\n"
        "- Company policies and handbooks\n"
        "- Benefits, compensation, and payroll\n"
        "- Onboarding and offboarding\n"
        "- Employee relations and conflict resolution\n"
        "- Compliance and labor law\n"
        "- Performance reviews and promotions\n"
        f"{shared_ctx}\n"
        "Output JSON: {'response': <your answer>, 'action_items': [list of next steps], "
        "'department': 'HR', 'confidence': 0.0-1.0}"
    )
    result = _call_llm_json(system, instruction)
    result.setdefault("department", "HR")
    result.setdefault("confidence", 0.8)

    memory.store("hr_agent", "last_response", result.get("response", ""))
    memory.store("hr_agent", "action_items", result.get("action_items", []))

    return result


def operations_agent(instruction: str) -> Dict[str, Any]:
    """
    Operations Department Agent.
    Handles: logistics, inventory, processes, supply chain, vendor management.
    """
    shared_ctx = _inject_shared_context()
    system = (
        "You are the OPERATIONS department AI agent. You handle inquiries about:\n"
        "- Supply chain and logistics\n"
        "- Inventory management and warehouse ops\n"
        "- Process optimization and SOPs\n"
        "- Vendor management and procurement\n"
        "- Quality assurance and audits\n"
        "- Capacity planning and resource allocation\n"
        f"{shared_ctx}\n"
        "Output JSON: {'response': <your answer>, 'action_items': [list of next steps], "
        "'department': 'Operations', 'confidence': 0.0-1.0}"
    )
    result = _call_llm_json(system, instruction)
    result.setdefault("department", "Operations")
    result.setdefault("confidence", 0.8)

    memory.store("operations_agent", "last_response", result.get("response", ""))
    memory.store("operations_agent", "action_items", result.get("action_items", []))

    return result


# ============================================================================
# LAYER 2: RECRUITMENT ANALYST AGENTS
# ============================================================================
def resume_analyzer(resume_text: str) -> Dict[str, Any]:
    """
    Resume Analyzer Agent.
    Extracts structured information from raw resume text:
    skills, experience, education, certifications, projects.
    Writes to shared memory so skills_matcher can read it.
    """
    shared_ctx = _inject_shared_context()
    system = (
        "You are an expert RESUME ANALYZER agent. Extract structured data from the resume.\n"
        f"{shared_ctx}\n"
        "Return JSON with these fields:\n"
        "{\n"
        '  "candidate_name": "string",\n'
        '  "email": "string or null",\n'
        '  "phone": "string or null",\n'
        '  "total_experience_years": number,\n'
        '  "current_role": "string",\n'
        '  "current_company": "string",\n'
        '  "technical_skills": ["list of technical skills"],\n'
        '  "soft_skills": ["list of soft skills"],\n'
        '  "certifications": ["list"],\n'
        '  "education": [{"degree": "...", "institution": "...", "year": "..."}],\n'
        '  "work_experience": [{"role": "...", "company": "...", "duration": "...", "highlights": ["..."]}],\n'
        '  "projects": ["list of notable projects"],\n'
        '  "summary": "2-3 sentence professional summary"\n'
        "}"
    )
    result = _call_llm_json(system, f"RESUME:\n\"\"\"\n{resume_text}\n\"\"\"")

    # Store EVERYTHING in shared memory — skills_matcher reads this
    memory.store("resume_analyzer", "candidate_name", result.get("candidate_name", "Unknown"))
    memory.store("resume_analyzer", "technical_skills", result.get("technical_skills", []))
    memory.store("resume_analyzer", "soft_skills", result.get("soft_skills", []))
    memory.store("resume_analyzer", "total_experience_years", result.get("total_experience_years", 0))
    memory.store("resume_analyzer", "education", result.get("education", []))
    memory.store("resume_analyzer", "work_experience", result.get("work_experience", []))
    memory.store("resume_analyzer", "certifications", result.get("certifications", []))
    memory.store("resume_analyzer", "projects", result.get("projects", []))
    memory.store("resume_analyzer", "summary", result.get("summary", ""))
    memory.store("resume_analyzer", "full_analysis", result)

    return result


def skills_matcher(jd_text: str) -> Dict[str, Any]:
    """
    Skills Matcher Agent.
    Reads resume analysis from shared memory + the JD text.
    Produces a match score, matched/missing/bonus skills, and a verdict.
    """
    # Read what resume_analyzer found
    resume_data = memory.recall_all_from_agent("resume_analyzer")
    if not resume_data:
        return {
            "error": "Resume has not been analyzed yet. Run resume_analyzer first.",
            "match_score": 0,
        }

    shared_ctx = _inject_shared_context()
    system = (
        "You are an expert SKILLS MATCHER agent. Compare the candidate's profile "
        "(from shared context) against the job description.\n"
        f"{shared_ctx}\n"
        "Return JSON:\n"
        "{\n"
        '  "match_score": 0-100,\n'
        '  "matched_skills": ["skills the candidate HAS that the JD requires"],\n'
        '  "missing_skills": ["skills the JD requires that the candidate LACKS"],\n'
        '  "bonus_skills": ["candidate skills not in JD but valuable"],\n'
        '  "experience_match": "over-qualified / well-matched / under-qualified",\n'
        '  "education_match": "meets requirements / partially meets / does not meet",\n'
        '  "strengths": ["top 3 candidate strengths for this role"],\n'
        '  "concerns": ["top 3 concerns or gaps"],\n'
        '  "verdict": "STRONG MATCH / MODERATE MATCH / WEAK MATCH",\n'
        '  "recommendation": "2-3 sentence hiring recommendation"\n'
        "}"
    )
    result = _call_llm_json(system, f"JOB DESCRIPTION:\n\"\"\"\n{jd_text}\n\"\"\"")

    # Store in shared memory — question_generator reads this
    memory.store("skills_matcher", "match_score", result.get("match_score", 0))
    memory.store("skills_matcher", "matched_skills", result.get("matched_skills", []))
    memory.store("skills_matcher", "missing_skills", result.get("missing_skills", []))
    memory.store("skills_matcher", "bonus_skills", result.get("bonus_skills", []))
    memory.store("skills_matcher", "verdict", result.get("verdict", "UNKNOWN"))
    memory.store("skills_matcher", "strengths", result.get("strengths", []))
    memory.store("skills_matcher", "concerns", result.get("concerns", []))
    memory.store("skills_matcher", "recommendation", result.get("recommendation", ""))
    memory.store("skills_matcher", "full_match", result)

    return result


def question_generator(jd_text: str) -> Dict[str, Any]:
    """
    Interview Question Generator Agent.
    Reads BOTH resume analysis AND skills match from shared memory.
    Generates targeted interview questions that probe:
        - Gaps identified by skills_matcher
        - Claims made in the resume
        - Role-specific scenarios from the JD
    """
    resume_data = memory.recall_all_from_agent("resume_analyzer")
    match_data = memory.recall_all_from_agent("skills_matcher")

    if not resume_data:
        return {"error": "Resume has not been analyzed yet."}
    if not match_data:
        return {"error": "Skills matching has not been done yet."}

    shared_ctx = _inject_shared_context()
    system = (
        "You are an expert INTERVIEW QUESTION GENERATOR. Using the candidate's resume analysis "
        "and skills match results from shared context, generate targeted interview questions.\n"
        f"{shared_ctx}\n"
        "Generate questions in these categories:\n"
        "1. TECHNICAL DEEP-DIVE: Probe the candidate's strongest claimed skills.\n"
        "2. GAP ASSESSMENT: Test the missing skills identified by the skills matcher.\n"
        "3. BEHAVIORAL (STAR): Situational questions based on their past roles.\n"
        "4. ROLE-SPECIFIC SCENARIO: Hypothetical problems they'd face in this new role.\n"
        "5. CULTURE FIT: Questions about collaboration, values, working style.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "technical_questions": [{"question": "...", "what_to_look_for": "...", "difficulty": "easy/medium/hard"}],\n'
        '  "gap_questions": [{"question": "...", "skill_being_tested": "...", "what_to_look_for": "..."}],\n'
        '  "behavioral_questions": [{"question": "...", "competency": "...", "what_to_look_for": "..."}],\n'
        '  "scenario_questions": [{"question": "...", "context": "...", "ideal_answer_highlights": "..."}],\n'
        '  "culture_fit_questions": [{"question": "...", "what_to_look_for": "..."}],\n'
        '  "recommended_interview_duration_minutes": number,\n'
        '  "overall_focus_areas": ["list of 3-5 key areas to focus on"]\n'
        "}\n"
        "Generate 3-4 questions per category."
    )
    result = _call_llm_json(system, f"JOB DESCRIPTION:\n\"\"\"\n{jd_text}\n\"\"\"")

    memory.store("question_generator", "questions", result)
    memory.store("question_generator", "focus_areas", result.get("overall_focus_areas", []))

    return result


# ============================================================================
# LAYER 3: VECTOR AGENT (Semantic Intelligence)
# ============================================================================
def vector_agent(text_a: str, text_b: str, context_label: str = "comparison") -> Dict[str, Any]:
    """
    Vector Agent — adds semantic intelligence to the system.

    Uses the LLM to perform embedding-style semantic analysis:
    - Measures conceptual similarity between two texts
    - Identifies semantic overlaps and gaps
    - Enriches shared context with similarity insights

    In a production system this would use real embeddings (OpenAI, Sentence-Transformers).
    Here we use the LLM as a semantic reasoner for the same effect without
    requiring a vector DB dependency.
    """
    shared_ctx = _inject_shared_context()
    system = (
        "You are a SEMANTIC ANALYSIS agent (Vector Agent). "
        "Analyze the conceptual similarity between two texts.\n"
        f"{shared_ctx}\n"
        "Return JSON:\n"
        "{\n"
        '  "similarity_score": 0.0-1.0,\n'
        '  "semantic_overlaps": ["list of shared concepts/themes"],\n'
        '  "unique_to_text_a": ["concepts only in text A"],\n'
        '  "unique_to_text_b": ["concepts only in text B"],\n'
        '  "alignment_summary": "2-3 sentence summary of how the texts relate",\n'
        '  "key_insights": ["3-5 insights from the comparison"]\n'
        "}"
    )
    user = f"TEXT A ({context_label} — first):\n\"\"\"\n{text_a}\n\"\"\"\n\nTEXT B ({context_label} — second):\n\"\"\"\n{text_b}\n\"\"\""
    result = _call_llm_json(system, user)

    memory.store("vector_agent", f"{context_label}_similarity", result.get("similarity_score", 0))
    memory.store("vector_agent", f"{context_label}_overlaps", result.get("semantic_overlaps", []))
    memory.store("vector_agent", f"{context_label}_insights", result.get("key_insights", []))
    memory.store("vector_agent", f"{context_label}_summary", result.get("alignment_summary", ""))

    return result
