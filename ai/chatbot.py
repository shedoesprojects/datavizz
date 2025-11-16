import os
from groq import Groq
import pandas as pd
import streamlit as st

# ------------------ Utility ------------------
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("⚠️ Groq API key not found. Please set GROQ_API_KEY in .env.")
    return Groq(api_key=api_key)


# ------------------ Chat Function ------------------
def chat_with_ai(message: str, df: pd.DataFrame = None, plot_choice: str = None):
    """
    Persistent conversational chatbot for DataWiz.
    Remembers past 10 messages via Streamlit session_state.
    Uses Groq LLaMA 3.3 models with fallback.
    """

    # Initialize persistent chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    df_summary = ""
    if df is not None:
        try:
            df_summary = df.describe(include="all").to_string()[:1200]
        except Exception:
            df_summary = f"Columns: {', '.join(df.columns[:10])}"

    # Compile last few exchanges for conversational context
    history_context = ""
    for chat in st.session_state.chat_history[-10:]:
        history_context += f"{chat['role'].capitalize()}: {chat['content']}\n"

    # Add user’s latest message to history
    st.session_state.chat_history.append({"role": "user", "content": message})

    # Build LLM prompt
    prompt = f"""
    You are DataWiz — a friendly and knowledgeable data visualization assistant.

    You help users interpret datasets, charts, and statistical concepts in clear, 
    simple language for beginners.

    Current chart: {plot_choice or "None selected"}
    Dataset summary (partial):
    {df_summary}

    Chat history:
    {history_context}

    Latest user question:
    {message}

    Respond in a friendly tone, with examples and insights relevant to the chart or dataset.
    If the question is general, give a concise and helpful data-related answer.
    """

    try:
        client = get_groq_client()

        # Primary model
        try:
            response = client.chat.completions.create(
                model="llama-3.3-8b-versatile",  # ✅ Current small model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400,
            )
        except Exception:
            # Fallback to large model
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400,
            )

        if response and response.choices:
            bot_reply = response.choices[0].message.content.strip()
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            return bot_reply

        return "⚠️ No response generated."

    except Exception as e:
        return f"⚠️ Chatbot error: {e}"
