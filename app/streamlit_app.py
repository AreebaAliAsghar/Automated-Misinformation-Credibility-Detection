import streamlit as st
from utils import get_prediction, get_category_style

st.set_page_config(
    page_title="Misinformation Credibility Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Automated Misinformation Credibility Detection")

st.markdown(
    """
    This system analyzes a news headline or claim and predicts a **LIAR truthfulness label**.
    It then converts the prediction into a **Credibility Score**, **Risk Score**, and
    **Final Credibility Category**.
    """
)

model_type = st.selectbox(
    "Choose Model:",
    ["Semantic ML Model", "Deep Learning Model"],
    help="Semantic ML uses Sentence-BERT embeddings + Softmax Logistic Regression. Deep Learning uses DistilBERT."
)

user_text = st.text_area(
    "Enter News Text Here:",
    height=150,
    placeholder="Example: Morocco wants tourists to visit Western Sahara. Some say it's tightening its control"
)

if st.button("Check Credibility"):
    if user_text.strip() == "":
        st.warning("Please enter some news text first.")

    else:
        with st.spinner("Analyzing credibility..."):
            result = get_prediction(user_text, model_type)

        category = result["final_category"]
        style = get_category_style(category)

        st.markdown("## Credibility Result")

        st.markdown(
            f"""
            <div style="
                background:{style['box_bg']};
                color:{style['box_text']};
                border:2px solid {style['border']};
                padding:18px;
                border-radius:14px;
                text-align:center;
                font-size:24px;
                font-weight:bold;
                margin-bottom:18px;
            ">
                Final Category: {category}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(f"Model Used: {result['model_used']}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted LIAR Label", result["predicted_label"])
            st.metric("Model Confidence", f"{result['model_confidence'] * 100:.2f}%")
            st.metric("Credibility Score", f"{result['credibility_score'] * 100:.2f}%")

        with col2:
            st.metric("Model Risk", f"{result['model_risk'] * 100:.2f}%")
            st.metric("Linguistic Risk", f"{result['linguistic_risk'] * 100:.2f}%")
            st.metric("Final Risk", f"{result['final_risk'] * 100:.2f}%")

        st.markdown("## Risk Meter")

        final_risk_percent = result["final_risk"] * 100

        st.markdown(
            f"""
            <div style="
                background:#e5e7eb;
                border-radius:999px;
                height:32px;
                width:100%;
                overflow:hidden;
                border:1px solid #d1d5db;
                margin-bottom:8px;
            ">
                <div style="
                    background:{style['bar']};
                    width:{final_risk_percent:.2f}%;
                    height:32px;
                    border-radius:999px;
                    text-align:center;
                    color:white;
                    font-weight:bold;
                    line-height:32px;
                    min-width:60px;
                ">
                    {final_risk_percent:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(result["final_risk"]))

        st.markdown("## Class Probabilities")

        sorted_probs = sorted(
            result["probability_dict"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for label, prob in sorted_probs:
            st.write(f"**{label}**: {prob * 100:.2f}%")
            st.progress(float(prob))

        st.markdown("## Linguistic Report")

        for reason in result["report"]:
            st.write("•", reason)

        st.markdown("## Interpretation")

        st.markdown(
            """
            The system uses a **soft risk scoring approach**, which means it does not only depend
            on the highest predicted label. It considers probabilities from all six LIAR classes:
            `true`, `mostly-true`, `half-true`, `barely-true`, `false`, and `pants-fire`.

            This gives a more stable credibility judgment for in-between cases.
            """
        )

st.markdown("---")

st.caption(
    "Built using NLP, Sentence-BERT embeddings, Softmax Logistic Regression, DistilBERT, PyTorch, and Streamlit."
)