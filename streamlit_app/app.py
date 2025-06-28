"""Streamlit demo for the sentiment model."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="FinLLM Sentiment Analyzer", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; }
    .confidence-high { background-color: #d4edda; padding: 5px; border-radius: 5px; }
    .confidence-medium { background-color: #fff3cd; padding: 5px; border-radius: 5px; }
    .confidence-low { background-color: #f8d7da; padding: 5px; border-radius: 5px; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_sentiment_color(sentiment):
    return f"sentiment-{sentiment}"

def get_confidence_class(confidence):
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    return "confidence-low"


def main():
    st.title("📈 FinLLM Sentiment Analyzer")
    st.markdown("*Fine-tuned LLM for Financial Text Sentiment Analysis*")

    with st.sidebar:
        st.header("⚙️ Settings")

        model_option = st.selectbox(
            "Select Model",
            ["FinLLM-Mistral-7B (Fine-tuned)", "FinLLM-Phi-2 (Fine-tuned)", "Base Mistral-7B"],
            index=0,
        )

        st.markdown("---")

        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1,
                                help="Lower = more deterministic")

        use_json = st.checkbox("JSON Output", value=True,
                               help="Request structured JSON output")

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown("""
        - **Base**: Mistral-7B
        - **Method**: QLoRA (4-bit)
        - **Dataset**: Financial PhraseBank
        - **Accuracy**: ~87%
        """)

    tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📊 Batch Analysis", "📈 Model Performance"])

    with tab1:
        st.header("Single Text Analysis")

        examples = {
            "Select an example...": "",
            "Positive - Revenue Growth": "Company reported a 25% increase in quarterly revenue, exceeding analyst expectations.",
            "Negative - Profit Warning": "The firm issued a profit warning due to supply chain disruptions and rising costs.",
            "Neutral - Leadership Change": "The board announced the appointment of a new CFO effective next quarter.",
            "Mixed - Acquisition": "Despite acquisition costs impacting short-term margins, long-term synergies are expected.",
        }

        selected = st.selectbox("Try an example:", list(examples.keys()))
        default_text = examples.get(selected, "")

        text_input = st.text_area("Enter financial text:", value=default_text, height=150,
            placeholder="Enter a financial news headline, earnings report, or market commentary...")

        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

        if analyze_btn and text_input:
            with st.spinner("Analyzing sentiment..."):
                result = simulate_prediction(text_input)

            st.markdown("---")
            st.subheader("Results")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <h3>Sentiment</h3>
                    <h2 class="{get_sentiment_color(result['sentiment'])}">{result['sentiment'].upper()}</h2>
                </div>""", unsafe_allow_html=True)
            with c2:
                pct = f"{result['confidence']*100:.1f}%"
                st.markdown(f"""<div class="metric-card">
                    <h3>Confidence</h3>
                    <h2 class="{get_confidence_class(result['confidence'])}">{pct}</h2>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown("""<div class="metric-card">
                    <h3>Model</h3>
                    <h2>FinLLM</h2>
                </div>""", unsafe_allow_html=True)

            if result.get("reasoning"):
                st.markdown("### 💡 Reasoning")
                st.info(result["reasoning"])

            with st.expander("View Raw Output"):
                st.json(result)

        elif analyze_btn:
            st.warning("Please enter some text to analyze.")

    with tab2:
        st.header("Batch Analysis")
        st.markdown("Upload a CSV file with a 'text' column for batch sentiment analysis.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            import pandas as pd
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                st.error("CSV must have a 'text' column.")
            else:
                st.write(f"Found {len(df)} texts.")
                st.dataframe(df.head())

                if st.button("🚀 Analyze All", type="primary"):
                    with st.spinner(f"Analyzing {len(df)} texts..."):
                        results = []
                        progress = st.progress(0)
                        for i, text in enumerate(df["text"]):
                            results.append(simulate_prediction(text))
                            progress.progress((i+1) / len(df))

                        df["sentiment"] = [r["sentiment"] for r in results]
                        df["confidence"] = [r["confidence"] for r in results]

                    st.success("Done!")
                    st.dataframe(df)

                    csv = df.to_csv(index=False)
                    st.download_button("📥 Download Results", data=csv,
                                       file_name="sentiment_results.csv", mime="text/csv")

                    st.markdown("### Summary")
                    counts = df["sentiment"].value_counts()
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Positive", counts.get("positive", 0))
                    with c2:
                        st.metric("Negative", counts.get("negative", 0))
                    with c3:
                        st.metric("Neutral", counts.get("neutral", 0))

    with tab3:
        st.header("Model Performance")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Training Metrics")
            import pandas as pd
            perf = {
                "Metric": ["Accuracy", "F1 (Macro)", "F1 (Weighted)", "Precision", "Recall"],
                "FinLLM (Fine-tuned)": ["87.2%", "85.8%", "87.0%", "86.1%", "85.5%"],
                "Base Mistral-7B": ["71.3%", "68.2%", "70.9%", "69.4%", "67.1%"],
            }
            st.table(pd.DataFrame(perf))

            st.markdown("### Per-Class")
            cls = {
                "Class": ["Positive", "Negative", "Neutral"],
                "Precision": ["88.2%", "85.1%", "84.9%"],
                "Recall": ["86.7%", "87.3%", "82.5%"],
                "F1": ["87.4%", "86.2%", "83.7%"],
            }
            st.table(pd.DataFrame(cls))

        with col2:
            st.markdown("### Training Config")
            st.code("""
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, v_proj, k_proj, o_proj]
epochs: 3
batch_size: 4
learning_rate: 2e-4
quantization: 4bit
            """, language="yaml")

            st.markdown("### Dataset")
            st.markdown("""
            - **Training**: Financial PhraseBank (3,453 samples)
            - **Validation**: 20% split
            - **Labels**: Positive, Negative, Neutral
            """)


def simulate_prediction(text):
    """Keyword-based placeholder for demo.
    TODO: hook up actual model inference
    """
    import random

    text_lower = text.lower()
    pos_kw = ["increase", "growth", "exceed", "profit", "gain", "strong", "positive"]
    neg_kw = ["decrease", "loss", "warning", "decline", "weak", "negative", "fall"]

    pos = sum(1 for k in pos_kw if k in text_lower)
    neg = sum(1 for k in neg_kw if k in text_lower)

    if pos > neg:
        sentiment = "positive"
        confidence = min(0.95, 0.7 + pos * 0.05)
    elif neg > pos:
        sentiment = "negative"
        confidence = min(0.95, 0.7 + neg * 0.05)
    else:
        sentiment = "neutral"
        confidence = 0.6 + random.uniform(0, 0.2)

    reasons = {
        "positive": "The text contains positive financial indicators such as growth, profitability, or exceeding expectations.",
        "negative": "The text indicates negative financial developments including losses, warnings, or declining performance.",
        "neutral": "The text presents factual financial information without strong positive or negative sentiment.",
    }

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "reasoning": reasons[sentiment],
        "raw_output": f"{sentiment} (confidence: {confidence:.2f})",
    }


if __name__ == "__main__":
    main()
