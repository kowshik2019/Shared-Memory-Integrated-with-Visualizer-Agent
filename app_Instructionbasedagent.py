"""
app.py — Streamlit UI for the Multi-Agent Orchestration System.

Two modes:
    1. Department Router — type an instruction, it routes to Sales/HR/Operations
    2. Recruitment Analyst — paste a resume + JD, get full pipeline analysis

Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from router_agent import router_agent

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Orchestration System",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Multi-Agent System")
    st.divider()
    st.markdown("**Architecture**")
    st.markdown(
        "1. 🚦 Router Agent — filters & classifies\n"
        "2. 🧮 Vector Agent — semantic analysis\n"
        "3. 🏢 Dept Agents — Sales / HR / Ops\n"
        "4. 📋 Recruitment Pipeline:\n"
        "   - Resume Analyzer\n"
        "   - Skills Matcher\n"
        "   - Question Generator"
    )
    st.divider()
    st.markdown("**Agent Communication**")
    st.caption(
        "All agents read/write to a shared memory store. "
        "Each agent sees what previous agents found, enabling "
        "collaborative intelligence."
    )
    st.divider()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY missing. Add it to `.env`.")
    st.caption("Model: " + os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Multi-Agent Orchestration System")
st.caption("Route instructions to the right department or run the full recruitment analysis pipeline.")

tab_dept, tab_recruit = st.tabs(["🏢 Department Router", "📋 Recruitment Analyst"])

# ===========================
# TAB 1: Department Router
# ===========================
with tab_dept:
    st.subheader("Route an Instruction")
    st.caption(
        "Type any business instruction. The Router Agent will filter it, "
        "classify it, and dispatch it to Sales, HR, or Operations."
    )

    dept_examples = {
        "— none —": "",
        "💰 Sales: pricing request":
            "What's our enterprise pricing for a 500-seat deployment? The client wants a 3-year deal.",
        "👥 HR: benefits inquiry":
            "What's our parental leave policy? A new hire is asking about maternity benefits.",
        "📦 Operations: inventory check":
            "We're running low on SKU-4432 in the Dallas warehouse. Can we get an ETA on the next shipment?",
        "🚫 Junk message (should be blocked)":
            "asdkjh asjdh kajshd random gibberish 12345!!!",
        "❓ Unclear message":
            "Can someone help me with the thing from last week?",
    }

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        dept_pick = st.selectbox(
            "Load an example (optional):",
            options=list(dept_examples.keys()),
            key="dept_example",
        )
        default_dept = dept_examples[dept_pick]
        dept_instruction = st.text_area(
            "Your instruction",
            value=default_dept,
            height=150,
            placeholder="Type a business instruction...",
            key="dept_input",
        )
        dept_submit = st.button("🚀 Route instruction", type="primary", use_container_width=True, key="dept_btn")

    with col_out:
        if dept_submit:
            if not dept_instruction.strip():
                st.warning("Please enter an instruction.")
            elif not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY missing.")
            else:
                with st.spinner("Router Agent processing..."):
                    try:
                        output = router_agent(instruction=dept_instruction)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.stop()

                status = output.get("status", "UNKNOWN")

                # Status badge
                if status == "BLOCKED":
                    st.error(f"🚫 BLOCKED — {output.get('reason', '')}")
                elif status == "UNCLEAR":
                    st.warning(f"❓ UNCLEAR — {output.get('message', '')}")
                elif status == "SUCCESS":
                    result = output.get("result", {})
                    classification = output.get("classification", {})

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Route", classification.get("route", "—").upper())
                    m2.metric("Confidence", f"{classification.get('confidence', 0):.0%}")
                    m3.metric("Status", status)

                    agent_out = result.get("agent_output", {})
                    st.markdown("### 💬 Agent Response")
                    st.info(agent_out.get("response", "No response generated."))

                    actions = agent_out.get("action_items", [])
                    if actions:
                        st.markdown("### ✅ Action Items")
                        for i, a in enumerate(actions, 1):
                            st.markdown(f"**{i}.** {a}")

                # Trace
                with st.expander("🔍 Routing Trace"):
                    for line in output.get("trace", []):
                        st.write("•", line)

                with st.expander("🧠 Shared Memory Snapshot"):
                    st.json(output.get("shared_memory", {}))
        else:
            st.info("Enter an instruction on the left and click Route.")


# ===========================
# TAB 2: Recruitment Analyst
# ===========================
with tab_recruit:
    st.subheader("Recruitment Analysis Pipeline")
    st.caption(
        "Paste a resume and a job description. Four agents will collaborate: "
        "Resume Analyzer → Vector Agent → Skills Matcher → Question Generator."
    )

    sample_resume = """John Smith
Senior Data Engineer | john.smith@email.com | (555) 123-4567

PROFESSIONAL SUMMARY
Experienced Data Engineer with 7+ years building scalable ETL pipelines,
data warehousing solutions, and real-time streaming architectures.

TECHNICAL SKILLS
Python, PySpark, SQL, Snowflake, dbt, Apache Airflow, Kafka, AWS (S3, Glue,
Redshift, Lambda), Docker, Kubernetes, Terraform, Git, CI/CD

CERTIFICATIONS
- AWS Certified Solutions Architect
- Google Cloud Professional Data Engineer
- Databricks Certified Associate

EXPERIENCE
Senior Data Engineer | TechCorp Inc. | 2021 - Present
- Designed real-time fraud detection pipeline processing 2M+ events/hour using Kafka and Spark
- Built and maintained 200+ dbt models across 15 business domains in Snowflake
- Reduced pipeline failures by 70% through idempotent design patterns and automated alerting
- Led migration of legacy Oracle warehouse to Snowflake, saving $400K/year in licensing

Data Engineer | DataFlow Solutions | 2018 - 2021
- Built batch ETL pipelines with Airflow orchestrating PySpark jobs on EMR
- Implemented CDC patterns for near-real-time data sync across 8 source systems
- Developed data quality framework catching 95% of anomalies before downstream consumption

Junior Developer | StartupXYZ | 2016 - 2018
- Built REST APIs in Python/Flask and integrated with PostgreSQL databases
- Automated deployment pipelines with Jenkins and Docker

EDUCATION
M.S. Computer Science | University of Texas at Arlington | 2016
B.Tech Information Technology | State University | 2014"""

    sample_jd = """Senior Data Engineer — FinTech Company (Remote)

We're looking for a Senior Data Engineer to build and scale our data platform.

REQUIREMENTS:
- 5+ years of experience in data engineering
- Strong proficiency in Python, SQL, and Spark/PySpark
- Experience with cloud data warehouses (Snowflake preferred, BigQuery acceptable)
- Hands-on experience with Apache Kafka or similar streaming platforms
- Experience with workflow orchestration (Airflow, Prefect, or Dagster)
- Knowledge of dbt for data transformation
- AWS or GCP cloud experience
- Experience with CI/CD pipelines and infrastructure-as-code (Terraform)

NICE TO HAVE:
- Experience in financial services or fintech
- Knowledge of machine learning pipelines
- Experience with Delta Lake or Apache Iceberg
- Real-time fraud detection or anomaly detection experience

RESPONSIBILITIES:
- Design and build scalable ETL/ELT pipelines processing 10M+ events daily
- Own the Snowflake data warehouse architecture including modeling and optimization
- Build real-time streaming solutions for transaction monitoring
- Implement data quality frameworks and monitoring
- Collaborate with ML engineers on feature engineering pipelines
- Mentor junior engineers and establish best practices"""

    col_resume, col_jd = st.columns(2, gap="medium")

    with col_resume:
        use_sample_resume = st.checkbox("Use sample resume", value=True, key="use_sample_resume")
        resume_input = st.text_area(
            "📄 Resume (paste text)",
            value=sample_resume if use_sample_resume else "",
            height=350,
            placeholder="Paste the candidate's resume here...",
            key="resume_input",
        )

    with col_jd:
        use_sample_jd = st.checkbox("Use sample JD", value=True, key="use_sample_jd")
        jd_input = st.text_area(
            "📋 Job Description",
            value=sample_jd if use_sample_jd else "",
            height=350,
            placeholder="Paste the job description here...",
            key="jd_input",
        )

    recruit_submit = st.button(
        "🚀 Run Recruitment Analysis Pipeline",
        type="primary",
        use_container_width=True,
        key="recruit_btn",
    )

    if recruit_submit:
        if not resume_input.strip() or not jd_input.strip():
            st.warning("Please provide both a resume and a job description.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY missing.")
        else:
            progress = st.progress(0, text="Starting recruitment pipeline...")

            try:
                # We run the full pipeline through the router
                with st.spinner(""):
                    progress.progress(10, text="Gate 1: Filtering message...")
                    output = router_agent(
                        instruction="Analyze this resume against the job description and generate interview questions.",
                        resume_text=resume_input,
                        jd_text=jd_input,
                    )
                progress.progress(100, text="Pipeline complete!")

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

            if output.get("status") != "SUCCESS":
                st.error(f"Pipeline failed: {output.get('message', output.get('reason', 'Unknown error'))}")
                st.stop()

            result = output.get("result", {})

            # ---- Pipeline Trace ----
            with st.expander("🔄 Pipeline Execution Trace", expanded=False):
                for line in output.get("trace", []):
                    st.write("•", line)

            # ---- Resume Analysis ----
            st.markdown("---")
            st.markdown("## 1. 📄 Resume Analysis")
            ra = result.get("resume_analysis", {})

            ra_m1, ra_m2, ra_m3 = st.columns(3)
            ra_m1.metric("Candidate", ra.get("candidate_name", "—"))
            ra_m2.metric("Experience", f"{ra.get('total_experience_years', '—')} years")
            ra_m3.metric("Current Role", ra.get("current_role", "—"))

            st.markdown(f"**Summary:** {ra.get('summary', '—')}")

            ra_c1, ra_c2 = st.columns(2)
            with ra_c1:
                st.markdown("**Technical Skills:**")
                tech = ra.get("technical_skills", [])
                if tech:
                    st.write(", ".join(tech))
            with ra_c2:
                st.markdown("**Certifications:**")
                certs = ra.get("certifications", [])
                if certs:
                    for c in certs:
                        st.write(f"• {c}")

            # ---- Vector Analysis ----
            st.markdown("---")
            st.markdown("## 2. 🧮 Vector / Semantic Analysis")
            va = result.get("vector_analysis", {})

            va_m1, va_m2 = st.columns(2)
            sim_score = va.get("similarity_score", 0)
            va_m1.metric("Semantic Similarity", f"{sim_score:.0%}" if isinstance(sim_score, (int, float)) else str(sim_score))
            va_m2.markdown(f"**Alignment:** {va.get('alignment_summary', '—')}")

            insights = va.get("key_insights", [])
            if insights:
                st.markdown("**Key Insights:**")
                for ins in insights:
                    st.write(f"• {ins}")

            # ---- Skills Match ----
            st.markdown("---")
            st.markdown("## 3. 🎯 Skills Match")
            sm = result.get("skills_match", {})

            sm_m1, sm_m2, sm_m3 = st.columns(3)
            sm_m1.metric("Match Score", f"{sm.get('match_score', 0)}/100")
            sm_m2.metric("Verdict", sm.get("verdict", "—"))
            sm_m3.metric("Experience Fit", sm.get("experience_match", "—"))

            sm_c1, sm_c2, sm_c3 = st.columns(3)
            with sm_c1:
                st.markdown("**✅ Matched Skills:**")
                for s in sm.get("matched_skills", []):
                    st.write(f"• {s}")
            with sm_c2:
                st.markdown("**❌ Missing Skills:**")
                for s in sm.get("missing_skills", []):
                    st.write(f"• {s}")
            with sm_c3:
                st.markdown("**⭐ Bonus Skills:**")
                for s in sm.get("bonus_skills", []):
                    st.write(f"• {s}")

            st.markdown(f"**Recommendation:** {sm.get('recommendation', '—')}")

            # ---- Interview Questions ----
            st.markdown("---")
            st.markdown("## 4. 🎤 Interview Questions")
            iq = result.get("interview_questions", {})

            q_categories = [
                ("🔧 Technical Deep-Dive", "technical_questions"),
                ("🔍 Gap Assessment", "gap_questions"),
                ("💡 Behavioral (STAR)", "behavioral_questions"),
                ("🎯 Role-Specific Scenarios", "scenario_questions"),
                ("🤝 Culture Fit", "culture_fit_questions"),
            ]

            for label, key in q_categories:
                questions = iq.get(key, [])
                if questions:
                    with st.expander(f"{label} ({len(questions)} questions)", expanded=False):
                        for i, q in enumerate(questions, 1):
                            if isinstance(q, dict):
                                st.markdown(f"**Q{i}: {q.get('question', '—')}**")
                                look_for = q.get("what_to_look_for", q.get("ideal_answer_highlights", ""))
                                if look_for:
                                    st.caption(f"What to look for: {look_for}")
                                difficulty = q.get("difficulty", q.get("competency", q.get("skill_being_tested", "")))
                                if difficulty:
                                    st.caption(f"Focus: {difficulty}")
                                st.markdown("---")
                            else:
                                st.markdown(f"**Q{i}:** {q}")

            focus = iq.get("overall_focus_areas", [])
            if focus:
                st.markdown("**🎯 Overall Focus Areas for Interview:**")
                for f in focus:
                    st.write(f"• {f}")

            duration = iq.get("recommended_interview_duration_minutes")
            if duration:
                st.info(f"⏱️ Recommended interview duration: {duration} minutes")

            # ---- Shared Memory ----
            with st.expander("🧠 Shared Memory — Full Agent Communication Log"):
                st.json(output.get("shared_memory", {}))
    else:
        st.info("Paste a resume and JD above (or use the samples), then click Run.")
