import streamlit as st
from io_utils.file_handler import read_uploaded_file
from plots.registry import list_plots, plot_metadata, generate_plot
from config import settings
import pandas as pd
import plots.core
from PIL import Image
import io
from ai.assistant import explain_plot, explain_concept
from ai.recommendation_engine import suggest_charts
from dotenv import load_dotenv
import os
from ai.chatbot import chat_with_ai
import os
from dotenv import load_dotenv

# Explicitly load the correct .env file
load_dotenv(dotenv_path="C:/Users/varsh/Desktop/datawiz/.env", override=True)

# Fallback: if .env doesn't load, use this line as backup
if not os.getenv("GROQ_API_KEY") or "your_actual_groq_key_here" in os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "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"  # your real key here

print("✅ Loaded GROQ key (debug):", os.getenv("GROQ_API_KEY")[:10], "...")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="DataWiz - Visualizer", layout="wide")
st.sidebar.title("📊 DataWiz — Upload & Plot")

# ------------------ FILE UPLOAD ------------------
uploaded = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xls", "xlsx"])
if uploaded:
    try:
        df = read_uploaded_file(uploaded)
        st.sidebar.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load file: {e}")
        st.stop()
else:
    st.info("📁 Upload a CSV / Excel file to begin.")
    st.stop()

# ------------------ LEARN SECTION ------------------
st.markdown("---")
st.header("🎓 Learn")

if "active_learn_section" not in st.session_state:
    st.session_state.active_learn_section = None

colA, colB, colC = st.columns(3)

with colA:
    if st.button("💡 Get Recommendations", use_container_width=True):
        st.session_state.active_learn_section = "recommendations"

with colB:
    if st.button("📘 Learn About a Chart", use_container_width=True):
        st.session_state.active_learn_section = "chart"

with colC:
    if st.button("🧠 Learn About a Concept", use_container_width=True):
        st.session_state.active_learn_section = "concept"

# --- Handle exclusive display ---
if st.session_state.active_learn_section == "recommendations":
    st.subheader("💡 Smart Chart Recommendations")
    try:
        with st.spinner("Analyzing your dataset..."):
            recommendations = suggest_charts(df)
        st.success(f"**Top Recommended Chart:** {recommendations['best']}")
        st.markdown("### 🥇 Top 3 Suggestions")
        for rec in recommendations["top3"]:
            st.markdown(f"**{rec['name']}** — {rec['reason']}")
        with st.expander("📊 See All Possible Charts", expanded=False):
            st.write(", ".join(recommendations["all_possible"]))
    except Exception as e:
        st.error(f"Couldn't generate recommendations: {e}")

elif st.session_state.active_learn_section == "chart":
    st.subheader("📘 Learn About a Chart")
    chart_choice = st.selectbox("Select a chart to learn about", list_plots())
    if chart_choice:
        plot_info = explain_plot(chart_choice)
        st.markdown(f"**What it shows:** {plot_info['explanation']}")
        with st.expander("💡 When to Use"):
            st.info(plot_info["use"])

elif st.session_state.active_learn_section == "concept":
    st.subheader("🧠 Learn About a Concept")
    concept_input = st.text_input("Enter a concept (e.g., correlation, variance)")
    if st.button("Explain Concept"):
        if concept_input.strip():
            st.write(explain_concept(concept_input))
        else:
            st.warning("Please enter a concept to learn about.")

# ------------------ PLOT CONFIG ------------------
plots = list_plots()
plot_choice = st.sidebar.selectbox("Plot Type", plots)
if "last_plot" not in st.session_state or st.session_state.last_plot != plot_choice:
    st.session_state.llm_output = None
    st.session_state.generate_llm = False
    st.session_state.last_plot = plot_choice

meta = plot_metadata(plot_choice)
params_needed = meta.get("params", [])

st.sidebar.markdown("---")
st.sidebar.subheader("Plot Options")

title = st.sidebar.text_input("Title", value="")
engine = st.sidebar.selectbox("Render Engine", options=meta.get("engines", ["matplotlib"]))
fig_w = st.sidebar.slider("Width (inches)", 4, 16, 8)
fig_h = st.sidebar.slider("Height (inches)", 3, 12, 4)

all_columns = list(df.columns)
numeric_columns = list(df.select_dtypes(include="number").columns)
text_columns = list(df.select_dtypes(exclude="number").columns)

params = {}
st.sidebar.markdown("---")
st.sidebar.subheader("Data Columns")

for p in params_needed:
    if p in ["x", "category", "task", "location", "path", "text", "source", "target",
             "date", "start", "finish", "category_col", "stage_col", "theta_col",
             "r_col", "x_col", "value_col", "date_col", "frame_col", "group"]:
        params[p] = st.sidebar.selectbox(f"{p.upper()}", options=all_columns, key=p)

    elif p in ["y", "value", "size", "open", "high", "low", "close", "lat", "lon",
               "error", "width", "r", "theta", "y_col", "z_col", "color_col",
               "size_col", "lat_col", "lon_col"]:
        params[p] = st.sidebar.selectbox(
            f"{p.upper()}",
            options=numeric_columns or all_columns,
            key=p
        )

    elif p == "hue":
        options = ["(None)"] + all_columns
        hue = st.sidebar.selectbox("HUE (optional)", options=options, key=p)
        params["hue"] = None if hue == "(None)" else hue

    elif p in ["y_cols", "vars", "columns", "dimension_columns", "path_cols", "values"]:
        params[p] = st.sidebar.multiselect(
            f"{p.upper()} (select multiple)",
            options=numeric_columns if p != "path_cols" else all_columns,
            default=numeric_columns[:min(3, len(numeric_columns))],
            key=p
        )

    elif p == "text":
        params[p] = st.sidebar.selectbox(f"Text Column", options=text_columns or all_columns, key=p)

    elif p == "bins":
        params[p] = st.sidebar.slider("Bins", 5, 100, 30, key=p)
    elif p == "lag":
        params[p] = st.sidebar.slider("Lag", 1, 50, 1, key=p)

    elif p == "column":
        params[p] = st.sidebar.selectbox(
            "Column (auto-select if empty)",
            options=["(Auto-select)"] + numeric_columns,
            key=p
        )
        if params[p] == "(Auto-select)":
            params[p] = None

    else:
        params[p] = st.sidebar.text_input(f"{p}", "", key=p)

if plot_choice == "Bar Chart":
    params["horizontal"] = st.sidebar.checkbox("Horizontal Bars", value=False)

params["title"] = title or uploaded.name.split('.')[0]
params["engine"] = engine
params["figsize"] = (fig_w, fig_h)

# --- 🎨 Customize Colors ---
with st.sidebar.expander("🎨 Customize Colors", expanded=False):
    theme_choice = st.selectbox(
        "Color Theme",
        ["Default", "Seaborn", "Plotly Dark", "Minimal", "Custom"],
        index=0,
        key="theme"
    )

    palette_choice = st.selectbox(
        "Palette",
        ["Default", "viridis", "plasma", "inferno", "magma", "cividis", "tab10", "Set2", "coolwarm"],
        index=0,
        key="palette"
    )

    custom_colors = {}
    if theme_choice == "Custom":
        st.markdown("#### Custom Colors")
        bg_color = st.color_picker("Background", "#FFFFFF", key="bg")
        text_color = st.color_picker("Text", "#000000", key="txt")
        plot_color = st.color_picker("Plot Elements (Bars/Lines/Points)", "#1f77b4", key="plot")
        custom_colors = {"background": bg_color, "text": text_color, "plot": plot_color}

    theme_config = {"theme": theme_choice, "palette": palette_choice, "custom_colors": custom_colors}
params["theme_config"] = theme_config

download_format = st.sidebar.selectbox("Choose Download Format", ["PNG", "PDF", "SVG"], index=0)

# ------------------ MAIN DISPLAY ------------------
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Visualization")
    with st.expander("Current Parameters", expanded=False):
        st.json({k: str(v) for k, v in params.items() if k not in ['figsize', 'title', 'engine']})

    if st.button("Render Plot", use_container_width=True):
        try:
            engine_used, payload = generate_plot(df, plot_choice, params.copy())
            if engine_used == "plotly":
                st.plotly_chart(payload, use_container_width=True)
                fmt = download_format.lower()
                data_bytes = payload.to_image(format=fmt)
                mime_map = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}
                st.download_button(f"Download {fmt.upper()}", data_bytes, file_name=f"{plot_choice}.{fmt}", mime=mime_map[fmt])
            else:
                st.image(payload, use_container_width=True)
                st.download_button("Download PNG", payload, file_name=f"{plot_choice}.png", mime="image/png")
        except Exception as e:
            st.error(f"❌ Failed to render: {e}")

# ==============================================
# 🧠 EXPLAIN THIS CHART — Auto LLM Integration
# ==============================================
from ai.groq_llm import explain_chart_with_llm

with st.expander("🧠 Explain This Chart", expanded=False):
    # --- Static educational info ---
    plot_info = explain_plot(plot_choice)
    st.markdown(f"**What this shows:** {plot_info['explanation']}")
    st.info(f"💡 **When to use:** {plot_info['use']}")

    # --- Groq dynamic explanation ---
    if GROQ_API_KEY:
        try:
            with st.spinner("Analyzing your chart using Groq AI..."):
                df_summary = df.describe(include="all").to_string()[:1500]
                llm_response = explain_chart_with_llm(plot_choice, df_summary)
                if llm_response:
                    st.success(llm_response)
                else:
                    st.warning("No AI explanation was generated.")
        except Exception as e:
            st.warning(f"⚠️ Groq LLM unavailable — using basic explanation. ({e})")
    else:
        st.info("Set your GROQ_API_KEY in .env to enable AI explanations.")

# ------------------ INSIGHTS PANEL ------------------
with col2:
    st.header("AI Insights")
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    if st.button("Generate Insights", use_container_width=True):
        if not settings.OPENAI_API_KEY:
            st.subheader("Quick Automatic Insights (Local)")
            st.markdown("### Summary Statistics")
            st.dataframe(df.describe().transpose(), use_container_width=True)
            st.markdown("### Missing Values")
            missing = df.isna().sum()
            missing_df = missing[missing > 0].sort_values(ascending=False)
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing values detected!")
            st.markdown("### Column Types")
            st.dataframe(pd.DataFrame({"Column": df.dtypes.index, "Type": df.dtypes.values.astype(str)}), use_container_width=True)
            numeric = df.select_dtypes(include="number")
            if numeric.shape[1] >= 2:
                st.markdown("### Top Correlations")
                corr = numeric.corr().abs().unstack().sort_values(ascending=False)
                corr = corr[corr < 1].drop_duplicates()
                st.dataframe(corr.head(10), use_container_width=True)
            else:
                st.info("Not enough numeric columns for correlation analysis.")
            if len(text_columns) > 0:
                st.markdown("### Categorical Distribution")
                cat_col = st.selectbox("Select Column", text_columns)
                value_counts = df[cat_col].value_counts().head(10)
                st.bar_chart(value_counts)
        else:
            st.info("AI insights via OpenAI integration coming soon!")
from ai.chatbot import chat_with_ai

with st.expander("💬 Ask DataWiz (AI Chatbot)", expanded=False):
    st.markdown("Chat with DataWiz about your dataset or visualization.")

    # Display previous messages
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for chat in st.session_state.chat_history[-8:]:
            role_icon = "🧍" if chat["role"] == "user" else "🧠"
            st.markdown(f"**{role_icon} {chat['role'].capitalize()}:** {chat['content']}")

    user_input = st.text_input("Ask your question:", key="chat_input")

    if st.button("Send"):
        if user_input.strip():
            reply = chat_with_ai(user_input, df, plot_choice)
            st.markdown(f"**🧠 DataWiz:** {reply}")
        else:
            st.warning("Please type a question before sending.")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_run()

# ------------------ SIDEBAR TIPS ------------------
st.sidebar.markdown("---")
with st.sidebar.expander("💡 Tips & Shortcuts"):
    st.markdown("""
    - **Multi-select:** Hold Ctrl/Cmd to select multiple columns  
    - **Auto-detect:** Some plots auto-select numeric columns  
    - **Hover:** Plotly charts are interactive  
    - **Download:** Choose format before exporting  
    """)

