# =====================================================
# RAG (Retrieval-Augmented Generation) - Streamlit
# =====================================================

import streamlit as st

from rag_utils import llm_caption, load_reviews, load_test_reviews, search, stream_llm

st.set_page_config(layout="wide")
st.title("🤖 RAG - Retrieval-Augmented Generation for Reviews")
st.markdown("---")

# =====================================================
# Load Data & pre-computed FAISS index
# =====================================================
df = load_reviews()
test_reviews = load_test_reviews()

st.header("📄 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)
st.caption(f"FAISS index: {len(df)} pre-computed review embeddings · {llm_caption()}")
st.markdown("---")


# =====================================================
# Retrieval & prompt
# =====================================================
def retrieve_similar_reviews(query, k=3):
    rows = df.iloc[search(query, k)]
    return [{"review_en": r.avis_en, "note": int(r.note)} for r in rows.itertuples()]


def generate_prompt(user_review, similar_reviews):
    context = "\n".join([
        f"- Rating: {r['note']}★\n  Review: {r['review_en']}"
        for r in similar_reviews
    ]) if similar_reviews else "No similar reviews found."

    return f"""
You are an expert in customer feedback writing.

Your task:
1. Correct spelling and grammar mistakes
2. Improve clarity and fluency
3. Keep the original meaning
4. Do NOT invent new information
5. Keep it natural and human

### Similar existing reviews:
{context}

### User review:
{user_review}

### Output format:
Corrected version:
...

Reformulated version:
...
"""


# =====================================================
# User Input / Test Review Selection
# =====================================================
st.header("🧪 Test a Single Review")

selected_review = st.selectbox("Select a review from the test dataset:", test_reviews)
user_input = st.text_area("Or enter your own review here:", "")
input_review = (user_input.strip() or selected_review or "").strip()

col1, col2 = st.columns(2)
preview_clicked = col1.button("Generate Prompt Preview")
predict_clicked = col2.button("Predict & Reformulate Review", type="primary")

if preview_clicked and input_review:
    similar = retrieve_similar_reviews(input_review)
    st.session_state["rag_for"] = input_review
    st.session_state["rag_similar"] = similar
    st.session_state["rag_prompt"] = generate_prompt(input_review, similar)

# The editable prompt stays visible (and keeps your edits) across reruns
# as long as the selected review does not change.
has_prompt = st.session_state.get("rag_for") == input_review and "rag_prompt" in st.session_state
if has_prompt:
    with st.expander("Show / Edit Prompt", expanded=True):
        st.text_area("Edit the prompt if you have specific requests:", key="rag_prompt", height=400)

# =====================================================
# Reformulate Review
# =====================================================
if predict_clicked:
    if not input_review:
        st.warning("Please enter a review or select one from the test dataset.")
    else:
        if has_prompt:
            similar = st.session_state["rag_similar"]
            prompt = st.session_state["rag_prompt"]
        else:
            similar = retrieve_similar_reviews(input_review)
            prompt = generate_prompt(input_review, similar)

        st.subheader("Original Review")
        st.write(input_review)

        st.subheader("Reformulated Review")
        try:
            st.write_stream(stream_llm(prompt))
        except Exception as e:
            st.error(f"LLM call failed: {e}")

        if similar:
            st.subheader("🔍 Similar Reviews Retrieved")
            for r in similar:
                st.markdown(f"- **{r['note']}★** | Review: {r['review_en']}")

st.markdown("---")
st.caption("RAG - Retrieval-Augmented Generation for review reformulation and rating context")
