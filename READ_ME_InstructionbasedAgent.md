# 🧠 Multi-Agent Orchestration System

A production-grade multi-agent AI system that routes business instructions to the right department (Sales / HR / Operations), blocks unwanted messages, and runs a full **Recruitment Analyst pipeline** where four collaborating agents analyze resumes, match skills against job descriptions, and generate targeted interview questions — all communicating through a shared memory store.

Built with **OpenAI**, **Streamlit**, and a custom **Router Agent** + **Shared Memory** architecture. No LangGraph or LangChain dependency — clean Python orchestration designed for clarity and extensibility.

---

## 📌 Project Description

### The Problem

Real organizations don't have a single AI agent that does everything. They have **departments** — Sales, HR, Operations — each with domain expertise. They have **recruitment workflows** where multiple steps depend on each other: you can't generate interview questions without first analyzing the resume, and you can't match skills without knowing the job requirements.

The challenge is **orchestration**: how do you route the right message to the right agent, prevent junk from wasting resources, and make agents share context so downstream agents benefit from upstream findings?

### The Solution

This system implements three architectural patterns:

**1. Router Agent Pattern**
A central Router Agent acts as the single entry point. Every message flows through it. The Router:
- **Filters** — blocks spam, gibberish, harmful content, and off-topic messages before they reach any department agent
- **Classifies** — determines whether the instruction belongs to Sales, HR, Operations, or the Recruitment pipeline
- **Dispatches** — calls the correct agent(s) in the correct sequence
- **Compiles** — assembles the final output with routing metadata and audit trail

**2. Shared Memory (Collaborative Memory)**
All agents read from and write to a thread-safe singleton memory store. This is how agents "communicate":
- Resume Analyzer writes `technical_skills: [Python, SQL, Spark]` to memory
- Skills Matcher reads that same list when comparing against JD requirements
- Question Generator reads BOTH the resume analysis AND the match results to ask targeted questions about gaps

No agent calls another agent directly. They communicate through the shared campfire — memory.

**3. Vector Agent (Semantic Intelligence)**
A specialized agent that performs semantic similarity analysis between texts (resume vs. JD). It identifies conceptual overlaps and gaps that keyword matching would miss, enriching the shared context before the Skills Matcher runs.

### What Makes This Different from a Single LLM Call

You *could* paste a resume + JD into ChatGPT and ask "analyze this." But:

- **A single call can't specialize.** The Resume Analyzer knows ONLY about extracting structured candidate data. The Skills Matcher knows ONLY about comparison logic. Specialization produces sharper output.
- **A single call has no memory.** Agent 3 can't reference what Agent 1 found. In this system, the Question Generator knows the candidate's exact skill gaps (from Skills Matcher) and their exact work history (from Resume Analyzer) and generates questions that probe both.
- **A single call can't filter.** The Router blocks garbage before it wastes LLM tokens. No department agent ever sees spam.
- **A single call can't route.** An HR question and a Sales question need different domain knowledge. The Router sends each to the right specialist.
- **A single call isn't auditable.** This system logs every agent's contribution to shared memory with timestamps, creating a full provenance trail.

---

## 🏗️ Architecture

### System Overview

```
                            ┌─────────────────────────────┐
                            │       ROUTER AGENT          │
                            │  (single entry point)       │
                            │                             │
                            │  1. Filter (block junk)     │
                            │  2. Classify (pick route)   │
                            │  3. Dispatch (call agents)  │
                            │  4. Compile (final output)  │
                            └──────────┬──────────────────┘
                                       │
                    ┌──────────────────┬┼─────────────────────┐
                    │                  │                       │
                    ▼                  ▼                       ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐
            │ SALES AGENT  │  │  HR AGENT    │  │   OPERATIONS AGENT       │
            │              │  │              │  │                          │
            │ pricing,     │  │ policies,    │  │ logistics, inventory,    │
            │ deals,       │  │ benefits,    │  │ supply chain,            │
            │ revenue      │  │ hiring       │  │ vendor management        │
            └──────────────┘  └──────────────┘  └──────────────────────────┘


            RECRUITMENT PIPELINE (when route = "recruitment"):

            ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
            │   RESUME     │────▶│   VECTOR     │────▶│   SKILLS     │────▶│  QUESTION    │
            │   ANALYZER   │     │   AGENT      │     │   MATCHER    │     │  GENERATOR   │
            │              │     │              │     │              │     │              │
            │ Extract:     │     │ Semantic     │     │ Compare:     │     │ Generate:    │
            │ - skills     │     │ similarity   │     │ - matched    │     │ - technical  │
            │ - experience │     │ between      │     │ - missing    │     │ - gap-based  │
            │ - education  │     │ resume & JD  │     │ - bonus      │     │ - behavioral │
            │ - certs      │     │              │     │ - verdict    │     │ - scenario   │
            └──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                   │                    │                    │                    │
                   ▼                    ▼                    ▼                    ▼
            ┌──────────────────────────────────────────────────────────────────────────┐
            │                        SHARED MEMORY                                     │
            │  (thread-safe singleton — every agent reads/writes here)                 │
            │                                                                          │
            │  resume_analyzer.technical_skills = [Python, SQL, Spark, ...]            │
            │  resume_analyzer.experience_years = 7                                    │
            │  vector_agent.similarity_score = 0.85                                    │
            │  skills_matcher.match_score = 88                                         │
            │  skills_matcher.missing_skills = [Delta Lake, ML pipelines]              │
            │  question_generator.questions = {technical: [...], gap: [...], ...}      │
            └──────────────────────────────────────────────────────────────────────────┘
```

### How Agents Communicate

The intelligence of this multi-agent system depends on communication between agents. Here's exactly how it works:

```
Step 1: Resume Analyzer runs
        → Writes to memory: skills, experience, education, certifications
        → Memory now has: [resume_analyzer.*]

Step 2: Vector Agent runs
        → Reads: nothing yet needed from memory (uses raw texts)
        → Writes to memory: similarity_score, overlaps, insights
        → Memory now has: [resume_analyzer.*, vector_agent.*]

Step 3: Skills Matcher runs
        → READS from memory: resume_analyzer.technical_skills, resume_analyzer.experience_years
        → READS from memory: vector_agent.similarity_score, vector_agent.insights
        → Uses all of this PLUS the JD to produce a match
        → Writes to memory: match_score, matched_skills, missing_skills, verdict
        → Memory now has: [resume_analyzer.*, vector_agent.*, skills_matcher.*]

Step 4: Question Generator runs
        → READS from memory: EVERYTHING above
        → Knows the candidate's exact skills (from resume_analyzer)
        → Knows exactly which skills are missing (from skills_matcher)
        → Knows the semantic alignment (from vector_agent)
        → Generates targeted questions that probe SPECIFIC gaps and claims
        → Writes to memory: questions, focus_areas
```

This is **collaborative memory** — each agent enriches the shared context, and downstream agents produce better output because they have more context. The Question Generator at the end is dramatically better than if it ran in isolation, because it knows exactly what to probe.

---

## 📁 Project Structure

```
multi_agent_system/
├── shared_memory.py      # Shared Memory store — the communication backbone
├── agents.py             # All agent definitions (department + recruitment + vector)
├── router_agent.py       # Router Agent — filter, classify, dispatch, compile
├── app.py                # Streamlit UI (two tabs)
├── requirements.txt      # Python dependencies
├── .env.example          # Template for environment variables
├── .env                  # Your API key (you create this; never commit)
└── README.md             # This file
```

---

## 📂 Code File Explanations

### `shared_memory.py` — The Communication Backbone

This file implements the `SharedMemory` class — a thread-safe singleton dictionary that all agents share. Think of it as the campfire where all agents gather to share what they've found.

**Key components:**

- **`MemoryEntry` dataclass** — A single fact stored by an agent, with fields for `agent` (who wrote it), `key` (what it's about), `value` (the data), and `timestamp` (when).

- **`SharedMemory` class** — Singleton pattern (only one instance exists, no matter how many times you import it). Uses `threading.Lock` for thread safety.

- **`store(agent, key, value)`** — How agents write. Example: `memory.store("resume_analyzer", "technical_skills", ["Python", "SQL"])`. Data is stored under a namespace (the agent name) so there are no key collisions.

- **`recall(agent, key)`** — How agents read a specific piece. Example: `skills = memory.recall("resume_analyzer", "technical_skills")`.

- **`recall_all_from_agent(agent)`** — Get everything one agent has stored. Used by downstream agents to see what upstream agents found.

- **`get_full_context()`** — Returns EVERYTHING from ALL agents. This is what gets injected into agent prompts as shared context.

- **`get_context_summary()`** — Human-readable string version of the full context. This gets embedded in LLM system prompts so agents are aware of each other's findings.

- **`get_history()`** — Chronological log of every `store()` call. Used for the audit trail in the UI.

- **`reset()`** — Clears all memory. Called at the start of each new request so agents start fresh.

- **`memory = SharedMemory()`** — Module-level singleton instance. Every file imports this same object.

### `agents.py` — All Agent Definitions

This file contains all seven agents organized in three layers. Every agent follows the same pattern: read shared context → do its job (LLM call) → write results to shared memory → return output.

**Helpers:**

- `_call_llm_json(system, user)` — Calls OpenAI in JSON mode with low temperature (0.3) for consistency. Wraps error handling.

- `_call_llm_text(system, user)` — Same but returns free-form text (available for agents that don't need structured output).

- `_inject_shared_context()` — Reads the shared memory and formats it as a string block that gets injected into every agent's system prompt. This is how agents "see" what other agents found.

**Layer 1 — Department Agents:**

- `sales_agent(instruction)` — Handles pricing, deals, revenue, pipeline, quotas. Has a Sales-specific system prompt. Writes its response and action items to shared memory.

- `hr_agent(instruction)` — Handles hiring policies, benefits, onboarding, compliance. Same pattern: specialized prompt, writes to memory.

- `operations_agent(instruction)` — Handles logistics, inventory, supply chain, vendor management. Same pattern.

All three department agents are structurally identical (they differ only in their domain-specific system prompts). This is intentional — it makes it trivial to add a fourth or fifth department.

**Layer 2 — Recruitment Agents:**

- `resume_analyzer(resume_text)` — The first agent in the recruitment pipeline. Takes raw resume text and extracts structured data: candidate name, skills, experience, education, certifications, work history, projects. Stores ALL of this in shared memory under the `resume_analyzer` namespace. This is the most memory-intensive agent — it writes 9 separate keys because downstream agents need different pieces.

- `skills_matcher(jd_text)` — The second (or third) agent. Reads the resume analysis from shared memory (it doesn't receive the resume text directly — it reads what `resume_analyzer` already extracted). Compares skills, experience, and education against the JD requirements. Produces a match score (0–100), lists of matched/missing/bonus skills, a verdict (STRONG/MODERATE/WEAK MATCH), and a hiring recommendation. Writes everything to memory for the question generator.

- `question_generator(jd_text)` — The final agent. Reads BOTH `resume_analyzer` and `skills_matcher` output from shared memory. Generates questions in five categories: Technical Deep-Dive, Gap Assessment, Behavioral (STAR), Role-Specific Scenarios, and Culture Fit. Each question includes guidance on what to look for in the answer. This agent produces the sharpest output in the system because it has the most context.

**Layer 3 — Vector Agent:**

- `vector_agent(text_a, text_b, context_label)` — Performs semantic similarity analysis between two texts. Returns a similarity score (0.0–1.0), lists of shared concepts, concepts unique to each text, and key insights. In the recruitment pipeline, it compares the resume against the JD before the Skills Matcher runs, enriching the shared context with semantic understanding that pure keyword matching would miss.

### `router_agent.py` — The Orchestrator

This is the brain. Every request enters through `router_agent()` and nothing bypasses it.

**Four steps in sequence:**

1. **`filter_message(instruction)`** — The first gate. Uses an LLM call to decide if the message is valid for processing. Blocks gibberish, spam, abusive content, off-topic nonsense, and injection attempts. Returns `is_valid` (boolean) and a reason. If blocked, the router returns immediately — no agent ever sees the message.

2. **`classify_instruction(instruction)`** — The second gate. Classifies the instruction into one of five routes: `sales`, `hr`, `operations`, `recruitment`, or `unclear`. Returns the route, a confidence score, and reasoning. If `unclear`, the router returns a clarification prompt listing the available departments.

3. **`orchestrate_department(route, instruction)`** — For department routes. Dispatches to the correct department agent (sales/hr/operations) and returns the result.

4. **`orchestrate_recruitment(resume_text, jd_text)`** — For the recruitment route. Runs the full four-agent pipeline in sequence: Resume Analyzer → Vector Agent → Skills Matcher → Question Generator. Builds a trace log showing what each agent found. Returns the combined output of all four agents plus a snapshot of shared memory.

**`router_agent(instruction, resume_text, jd_text)`** — The main entry point. Resets shared memory, runs the filter, runs the classifier, dispatches to the right orchestration function, and compiles the final output with status, routing metadata, agent results, shared memory snapshot, and full trace.

### `app.py` — The Streamlit UI

Two-tab interface:

**Tab 1 — Department Router:** A text input for any business instruction, an example dropdown (including a spam example to demonstrate blocking), and the result panel showing the routed department, confidence, agent response, and action items.

**Tab 2 — Recruitment Analyst:** Side-by-side text areas for resume and JD (pre-loaded with realistic samples), a progress bar, and four result sections — Resume Analysis (metrics + skills), Vector Analysis (similarity score + insights), Skills Match (score/verdict/matched/missing), and Interview Questions (5 expandable categories with look-for guidance). At the bottom, a shared memory expander shows the full agent communication log.

---

## ⚙️ Prerequisites

1. **Python 3.10 or newer** — check with `python --version`
2. **An OpenAI API key** with available credits — get one from https://platform.openai.com/api-keys
3. **Billing configured** — verify at https://platform.openai.com/settings/organization/billing
4. **A code editor** — Visual Studio Code recommended
5. **A terminal** — PowerShell (Windows), Terminal (Mac/Linux), or VS Code integrated terminal

---

## 🚀 Step-by-Step Setup & Execution

### Step 1 — Get the Project Files

Place all files into a folder named `multi_agent_system`:

```
multi_agent_system/
├── shared_memory.py
├── agents.py
├── router_agent.py
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

### Step 2 — Open a Terminal in the Project Folder

```bash
cd path/to/multi_agent_system
```

### Step 3 — Create a Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`.

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Configure Your API Key

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open `.env` in your editor and replace the placeholder:

```
OPENAI_API_KEY=sk-proj-your-real-key-here
OPENAI_MODEL=gpt-4o-mini
```

Rules:
- No quotes around the key
- No spaces around `=`
- Entire key on ONE line
- File named exactly `.env` (not `.env.txt`)

### Step 6 — Launch the Application

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

### Step 7 — Test the Department Router

1. Click the **🏢 Department Router** tab
2. Select "💰 Sales: pricing request" from the dropdown
3. Click **🚀 Route instruction**
4. See the Router classify it as SALES, dispatch to the Sales Agent, and return a response with action items
5. Try the "🚫 Junk message" example to see the filter block it

### Step 8 — Test the Recruitment Pipeline

1. Click the **📋 Recruitment Analyst** tab
2. Leave "Use sample resume" and "Use sample JD" checked (realistic pre-loaded data)
3. Click **🚀 Run Recruitment Analysis Pipeline**
4. Watch four agents run in sequence:
   - Resume Analysis section appears (candidate profile, skills extracted)
   - Vector Analysis section appears (semantic similarity score)
   - Skills Match section appears (match score, matched/missing skills, verdict)
   - Interview Questions section appears (5 categories of targeted questions)
5. Open the **🧠 Shared Memory** expander at the bottom to see the full agent communication log

### Step 9 — Try Your Own Data

Uncheck the sample checkboxes, paste a real resume and real JD, and run the pipeline. The agents will collaborate on your actual data.

### Step 10 — Stop the App

Press **Ctrl+C** in the terminal. To exit the virtual environment: `deactivate`.

---

## 🧪 Test Scenarios

### Department Router Tests

| Instruction | Expected Route | Expected Result |
|---|---|---|
| "What's our enterprise pricing for 500 seats?" | SALES | Pricing breakdown + next steps |
| "What's our parental leave policy?" | HR | Policy summary + action items |
| "We're low on SKU-4432 in Dallas warehouse" | OPERATIONS | Inventory check + procurement steps |
| "asdkjh asjdh random gibberish" | BLOCKED | Message filtered before reaching any agent |
| "Can someone help with the thing from last week?" | UNCLEAR | Clarification prompt listing departments |

### Recruitment Pipeline Tests

| Scenario | What to Observe |
|---|---|
| Strong candidate (sample data) | High match score (80+), few missing skills, STRONG MATCH verdict |
| Paste a marketing resume against data eng JD | Low match score, many missing skills, gap questions targeting technical holes |
| Very short resume (just a name and one skill) | Resume Analyzer extracts what it can; Skills Matcher flags many missing items |
| No resume provided | Router returns NEEDS_INPUT status with a helpful message |

---

## 🔧 Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any chat-completions model. `gpt-4o-mini` is fast and cheap; `gpt-4o` produces higher quality for ambiguous inputs |

---

## 🛠️ Extending the System

### Add a new department

1. Write a new function in `agents.py` (copy `sales_agent` as template, change the system prompt)
2. Add it to the `dispatch` dict in `router_agent.py::orchestrate_department`
3. Add the route to the classifier prompt in `router_agent.py::classify_instruction`

### Add a real vector database

Replace the LLM-based `vector_agent` with actual embeddings:
- Use `openai.embeddings.create()` to embed resume and JD
- Store in ChromaDB or FAISS
- Compute cosine similarity
- The rest of the pipeline stays the same — just write results to shared memory

### Add PDF resume upload

Install `pdfplumber` or `PyPDF2`, add a file uploader in Streamlit, extract text, and pass it to the router.

### Add multi-model routing

Route simple department queries to `gpt-4o-mini` (cheap) and complex recruitment analysis to `gpt-4o` (more capable). Change the `OPENAI_MODEL` per-agent in `agents.py`.

### Persist results to database

Add a SQLite write in `router_agent()` after compilation. Store the full output dict for audit trail and analytics.

---

## 🐛 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY missing` in sidebar | `.env` file missing or misnamed | Create `.env` (not `.env.txt`) with your key |
| `401 Incorrect API key` | Key revoked, mistyped, or no billing | Generate new key, replace in `.env`, restart Streamlit |
| Old key used after editing `.env` | Streamlit cached old env | Ctrl+C, restart `streamlit run app.py` |
| Shell env overriding `.env` | `OPENAI_API_KEY` set in shell/system env | Code uses `override=True` so `.env` wins; if still failing, unset the shell variable |
| `ModuleNotFoundError` | Dependencies not installed or wrong venv | Activate venv, re-run `pip install -r requirements.txt` |
| Skills Matcher says "Resume has not been analyzed" | Agents ran out of order | This shouldn't happen through the router; if using agents directly, call `resume_analyzer` first |
| All questions are generic | Shared memory didn't propagate | Check the memory expander in UI; if empty, resume_analyzer may have failed silently |

---

## 📚 Tech Stack

| Technology | Role |
|---|---|
| **[OpenAI Python SDK](https://github.com/openai/openai-python)** | Powers every agent via `gpt-4o-mini` in JSON mode |
| **[Streamlit](https://streamlit.io/)** | Two-tab interactive UI with metrics, expanders, and progress bars |
| **[python-dotenv](https://github.com/theskumar/python-dotenv)** | Loads API key from `.env` with `override=True` for safety |
| **Custom SharedMemory** | Thread-safe singleton store enabling inter-agent communication |
| **Custom Router Agent** | Filter → Classify → Dispatch → Compile orchestration pattern |

No LangGraph or LangChain dependency. The orchestration is pure Python, making it easy to understand, debug, and extend.

---

## 📝 License & Production Readiness

This is a learning and demonstration project. Before deploying to production, add:

- Rate limiting on OpenAI calls (per-user and global)
- Authentication on the Streamlit UI
- Input sanitization and prompt-injection guardrails
- Logging and observability (e.g., Langfuse, OpenTelemetry)
- Ticket/result persistence (database)
- Cost monitoring and per-request token tracking
- Error retry logic with exponential backoff
- Unit tests for each agent and the router
- Load testing for concurrent requests (SharedMemory is thread-safe but the OpenAI calls are serial)

---

*Built with care. Agents are strongest when they collaborate.* 🧠🤝
