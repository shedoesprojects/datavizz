"""AI-less smart chart recommender for DataWiz."""

import numpy as np
import pandas as pd


def suggest_charts(df: pd.DataFrame) -> dict:
    """
    Analyze dataframe structure and suggest top and possible visualizations.
    Returns a dict with 'best', 'top3', and 'all_possible' keys.
    """

    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(exclude="number").columns)
    datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    text_cols = [c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.len().mean() > 15]

    all_suggestions = []
    scored = []

    # --- Rules ---

    if len(numeric_cols) >= 2:
        all_suggestions += ["Scatter Plot", "Pair Plot", "Heatmap (Correlation)", "Bubble Chart"]
        scored.append(("Scatter Plot", 0.95, "Detected 2+ numeric columns suitable for x/y comparison."))
        scored.append(("Heatmap (Correlation)", 0.9, "Multiple numeric columns — correlation matrix possible."))

    if len(numeric_cols) == 1 and len(categorical_cols) >= 1:
        all_suggestions += ["Bar Chart", "Box Plot", "Violin Plot", "Strip Plot"]
        scored.append(("Bar Chart", 0.9, "One numeric and one categorical column — ideal for comparing groups."))

    if len(numeric_cols) == 1 and not categorical_cols:
        all_suggestions += ["Histogram", "Density Contour"]
        scored.append(("Histogram", 0.85, "Single numeric column — suitable for distribution visualization."))

    if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
        all_suggestions += ["Line Plot", "Time Series", "Area Chart"]
        scored.append(("Time Series", 0.95, "Detected datetime and numeric column — fits time trend plots."))

    if len(categorical_cols) >= 2:
        all_suggestions += ["Grouped Bar Chart", "Stacked Bar Chart", "Treemap", "Sunburst"]
        scored.append(("Grouped Bar Chart", 0.85, "2+ categorical columns — grouped comparisons are informative."))

    if text_cols:
        all_suggestions += ["Word Cloud"]
        scored.append(("Word Cloud", 0.9, "Textual column detected — word frequency visualization possible."))

    if {"lat", "lon"}.issubset(df.columns):
        all_suggestions += ["Scatter Geo Map"]
        scored.append(("Scatter Geo Map", 0.95, "Latitude/Longitude detected — ideal for geospatial mapping."))

    if "country" in df.columns or "Country" in df.columns:
        all_suggestions += ["Choropleth Map"]
        scored.append(("Choropleth Map", 0.9, "Country column detected — suitable for choropleth map."))

    if {"source", "target", "value"}.issubset(df.columns):
        all_suggestions += ["Sankey Diagram", "Network Graph"]
        scored.append(("Sankey Diagram", 0.9, "Source → Target relationships found — ideal for flow diagrams."))

    if {"task", "start", "finish"}.issubset(df.columns):
        all_suggestions += ["Gantt Chart"]
        scored.append(("Gantt Chart", 0.9, "Task schedule data detected — fits Gantt timeline visualization."))

    if len(numeric_cols) >= 3:
        all_suggestions += ["3D Scatter Plot", "Radar Chart", "Parallel Coordinates"]
        scored.append(("3D Scatter Plot", 0.9, "3+ numeric columns allow 3D and multidimensional plots."))

    # Deduplicate list preserving order
    all_suggestions = list(dict.fromkeys(all_suggestions))

    # Pick top 3 with highest confidence
    top3_sorted = sorted(scored, key=lambda x: x[1], reverse=True)[:3]
    top3 = [{"name": t[0], "reason": t[2]} for t in top3_sorted]
    best = top3[0]["name"] if top3 else (all_suggestions[0] if all_suggestions else "No suitable plots found")

    return {
        "best": best,
        "top3": top3,
        "all_possible": all_suggestions,
    }
