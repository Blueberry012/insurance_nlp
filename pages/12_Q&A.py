# =====================================================
# Q&A for Insurance Reviews - Streamlit
# =====================================================

import streamlit as st

from rag_utils import llm_caption, load_reviews, search, stream_llm

st.set_page_config(layout="wide")
st.title("🧠 Q&A - Insurance Reviews")
st.markdown("---")

# =====================================================
# Load Data & pre-computed FAISS index
# =====================================================
df = load_reviews()
known_insurers = sorted(df["assureur"].unique(), key=len, reverse=True)  # longest match first

st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.caption(f"FAISS index: {len(df)} pre-computed review embeddings · {llm_caption()}")
st.markdown("---")


# =====================================================
# Extract insurer & category
# =====================================================
def extract_insurer(question):
    question_lower = question.lower()
    for insurer in known_insurers:
        if insurer.lower() in question_lower:
            return insurer
    return None


def extract_category(question):
    question_lower = question.lower()
    category_keywords = {
        "pricing": ["price", "pricing", "cost", "expensive", "cheap", "tarif"],
        "customer_service": ["service", "support", "customer"],
        "claims": ["claim", "refund", "reimbursement"],
        "delay": ["delay", "slow", "waiting"],
        "contract": ["contract", "policy"],
    }
    for category, keywords in category_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    return None


# =====================================================
# Retrieval
# =====================================================
def retrieve_relevant_reviews(question, k=5):
    insurer = extract_insurer(question)
    category = extract_category(question)
    query = f"{question} about {category}" if category else question

    # When filtering on an insurer, search wider so that k reviews survive the filter
    candidates = df.iloc[search(query, 300 if insurer else k)]
    if insurer:
        candidates = candidates[candidates["assureur"].str.lower() == insurer.lower()]

    return [
        {"review_en": r.avis_en, "note": int(r.note), "assureur": r.assureur}
        for r in candidates.head(k).itertuples()
    ]


def generate_qa_prompt(question, reviews):
    context = "\n".join([
        f"""Review {i+1} (Insurer: {r['assureur']}, Rating: {r['note']}★):
Text: {r['review_en']}"""
        for i, r in enumerate(reviews)
    ])
    return f"""
You are an expert insurance analyst.

### Reviews:
{context}

### Question:
{question}

### Instructions:
- Answer in 2-3 sentences
- Add bullet points if relevant
- Mention insurer(s)

### Answer:
"""


# =====================================================
# Streamlit UI with editable prompt
# =====================================================
st.header("📝 Ask a Question")

predefined_questions = [
    "What do customers say about AXA claims?",
    "What are common customer service issues?",
    "What are pricing complaints?",
    "Is AXA expensive?",
    "Are there delays in reimbursements?",
]

selected_question = st.selectbox("Or choose a question from test set:", predefined_questions)
user_question = st.text_area("Or enter your own question here:", "")
question_to_ask = user_question.strip() or selected_question

col1, col2 = st.columns(2)
preview_clicked = col1.button("Generate Prompt Preview")
ask_clicked = col2.button("Ask Question", type="primary")

if preview_clicked and question_to_ask:
    reviews = retrieve_relevant_reviews(question_to_ask)
    st.session_state["qa_for"] = question_to_ask
    st.session_state["qa_reviews"] = reviews
    st.session_state["qa_prompt"] = generate_qa_prompt(question_to_ask, reviews)

# The editable prompt stays visible (and keeps your edits) across reruns
# as long as the question does not change.
has_prompt = st.session_state.get("qa_for") == question_to_ask and "qa_prompt" in st.session_state
if has_prompt:
    with st.expander("Show / Edit Prompt", expanded=True):
        st.text_area("Edit the prompt if you have specific instructions:", key="qa_prompt", height=400)

# =====================================================
# Ask Question with optional edited prompt
# =====================================================
if ask_clicked:
    if not question_to_ask:
        st.warning("Please enter a question or select one from the test set.")
    else:
        if has_prompt:
            reviews = st.session_state["qa_reviews"]
            prompt = st.session_state["qa_prompt"]
        else:
            reviews = retrieve_relevant_reviews(question_to_ask)
            prompt = generate_qa_prompt(question_to_ask, reviews)

        if not reviews:
            st.info("No relevant reviews found.")
        else:
            st.subheader("✅ Answer")
            try:
                st.write_stream(stream_llm(prompt))
            except Exception as e:
                st.error(f"LLM call failed: {e}")

            st.subheader("🔍 Relevant Reviews:")
            for i, r in enumerate(reviews):
                st.markdown(f"**Review {i+1}** (Insurer: {r['assureur']}, Rating: {r['note']}★)")
                st.markdown(f"- Text: {r['review_en']}")

st.markdown("---")
st.caption("Q&A - Insurance Review Insights (powered by RAG)")
