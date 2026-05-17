# Multi Agent Orchestration & Research Analyst Using Streamlit UI


# Worflow for the agent Recruitment Analyst
Resume Analyzer writes → shared_memory ← Skills Matcher reads
                                        ← Question Generator reads EVERYTHING

No agent calls another agent directly. They communicate through memory. That's why the Question Generator produces targeted questions about specific gaps — it knows what Skills Matcher found missing because it reads the same campfire.

To Execute:

cd FileName
pip install -r requirements.txt
streamlit run app_instructionbasedAagent.py



