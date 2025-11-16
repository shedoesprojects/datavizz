import os
from groq import Groq

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("⚠️ Groq API key not found. Please set GROQ_API_KEY in .env.")
    return Groq(api_key=api_key)


def explain_chart_with_llm(plot_name: str, df_summary: str):
    """
    Generate a beginner-friendly explanation of the chart and dataset using Groq LLM.
    Uses the latest (Nov 2025) LLaMA 3.3 models with fallback.
    """
    try:
        client = get_groq_client()
        prompt = f"""
        You are DataWiz — a data visualization mentor for beginners.

        Chart type: {plot_name}
        Dataset summary (partial):
        {df_summary}

        Explain clearly:
        1. What this chart type represents.
        2. What patterns the dataset might show.
        3. One or two real-world scenarios where it’s useful.

        Keep it short, easy to read, and under 200 words.
        """

        # ✅ Primary model
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=300,
            )
        except Exception:
            # ✅ Backup model (smaller)
            response = client.chat.completions.create(
                model="llama-3.3-8b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=300,
            )

        if response and response.choices:
            return response.choices[0].message.content.strip()

        return "⚠️ No explanation generated."

    except Exception as e:
        return f"⚠️ Error connecting to Groq: {e}"
