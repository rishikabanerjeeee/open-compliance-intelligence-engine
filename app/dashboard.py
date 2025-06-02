import streamlit as st
import pandas as pd
from utils import plot_utils
from models import match_engine

st.set_page_config(page_title="Compliance Intelligence Engine", layout="wide")
st.title("🛡️ Compliance Intelligence Engine")

# Sidebar Options
st.sidebar.header("🔍 Navigation")
option = st.sidebar.radio("Choose a module:", ["Upload & Match", "Compliance Score", "Visualizations", "Chatbot"])

# Shared session state
if "match_results" not in st.session_state:
    st.session_state.match_results = []

# --- 1. Upload & Match ---
if option == "Upload & Match":
    st.subheader("📄 Upload Policy Controls")

    uploaded_file = st.file_uploader("Upload your policy controls (JSON)", type=["json"])
    if uploaded_file:
        policy_data = pd.read_json(uploaded_file)
        st.json(policy_data.to_dict())

        st.info("Matching with regulations...")

        # Load regulations from a local JSON
        reg_data = match_engine.load_regulations("data/regulations.json")
        matches = match_engine.get_matches(policy_data.to_dict(orient="records"), reg_data)

        st.session_state.match_results = matches
        st.success("✅ Matching complete. Now check the next tabs!")

# --- 2. Compliance Score ---
elif option == "Compliance Score":
    st.subheader("📊 Compliance Score & Gap Analysis")

    if not st.session_state.match_results:
        st.warning("Please run matching first.")
    else:
        regulations = match_engine.load_regulations("data/regulations.json")
        total_regs = len(regulations)
        scores, coverage = match_engine.compute_compliance_score(st.session_state.match_results, total_regs)

        df = pd.DataFrame({
            "Control": [f"Control {i+1}" for i in range(len(scores))],
            "Compliance Score (%)": [round(s * 100, 2) for s in scores],
        })
        st.dataframe(df)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="compliance_score.csv")

# --- 3. Visualizations ---
elif option == "Visualizations":
    st.subheader("📈 Visualizations")

    if not st.session_state.match_results:
        st.warning("Please run matching first.")
    else:
        # Pass compliance score to plotting functions
        regulations = match_engine.load_regulations("data/regulations.json")
        total_regs = len(regulations)
        scores, _ = match_engine.compute_compliance_score(st.session_state.match_results, total_regs)

        st.pyplot(plot_utils.plot_bar_chart(scores))
        st.pyplot(plot_utils.plot_pie_chart(st.session_state.match_results))
        st.pyplot(plot_utils.plot_heatmap(st.session_state.match_results))

# --- 4. Chatbot ---
elif option == "Chatbot":
    st.subheader("💬 Compliance Chatbot")

    user_input = st.text_input("Ask a compliance-related question:")
    if user_input:
        # Mock or basic QA based on matches
        st.write("🤖 (Bot): This feature is coming soon!")
