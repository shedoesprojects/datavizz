"""
AI Companion module for DataWiz
Explains all 107 DataWiz plot types in plain English + 'When to Use' tips.
"""

import pandas as pd
from dotenv import load_dotenv
import os
from ai.groq_llm import explain_chart_with_llm
import pandas as pd
load_dotenv()  # loads .env variables automatically

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------------
# 🎯 SMART GRAPH SUGGESTION
# ------------------------------
def suggest_graph(df: pd.DataFrame):
    """Suggest an ideal graph type based on data composition."""
    num_cols = list(df.select_dtypes(include='number').columns)
    cat_cols = list(df.select_dtypes(exclude='number').columns)

    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        return (
            "Bar Chart",
            "Your data has both categories and numbers — try a Bar or Box Plot to compare them easily.",
        )
    elif len(num_cols) >= 2:
        return (
            "Scatter Plot",
            "Two or more numeric columns detected — a Scatter or Line Plot can reveal relationships or trends.",
        )
    elif any("date" in c.lower() for c in df.columns):
        return (
            "Time Series",
            "Date or time column found — visualize changes over time using Line or Area Charts.",
        )
    elif len(num_cols) == 1:
        return (
            "Histogram",
            "Single numeric column — visualize its distribution using a Histogram or Violin Plot.",
        )
    else:
        return (
            "Pie Chart",
            "Categorical data only — Pie or Donut Chart best shows proportions or shares.",
        )


# ------------------------------
# 📖 GRAPH EXPLANATIONS
# ------------------------------
def explain_plot(plot_type: str, df: pd.DataFrame = None):
    """
    Returns a rich explanation of the chart type.
    Uses Groq LLM if available and .env key is set,
    otherwise falls back to predefined text.
    """
    info = {
        "Bar Chart": {
            "explanation": "Compares quantities across categories using rectangular bars. Taller bars mean higher values.",
            "use": "Use when comparing discrete categories or groups side by side.",
        },
        "Stacked Bar Chart": {
            "explanation": "Displays how parts make up a whole — each section stacked within the same bar.",
            "use": "Use when showing how categories contribute to a total.",
        },
        "Grouped Bar Chart": {
            "explanation": "Places bars for multiple groups side by side for direct comparison.",
            "use": "Use when comparing subcategories across main categories.",
        },
        "Line Plot": {
            "explanation": "Connects data points with lines to show changes or trends over time.",
            "use": "Use when tracking progress or variation across a continuous variable.",
        },
        "Scatter Plot": {
            "explanation": "Shows how two numeric variables relate by plotting dots.",
            "use": "Use to observe correlation or clustering between variables.",
        },
        "Area Chart": {
            "explanation": "Fills the space under a line chart to emphasize volume or totals over time.",
            "use": "Use when you want to show how quantities evolve cumulatively.",
        },
        "Pie Chart": {
            "explanation": "Represents parts of a whole using slices of a circle.",
            "use": "Use for showing proportions or percentages within one total.",
        },
        "Donut Chart": {
            "explanation": "A Pie Chart with a hole in the center, making labels easier to read.",
            "use": "Use for proportional comparison with a modern visual style.",
        },
        "Histogram": {
            "explanation": "Shows frequency of data within ranges using bars.",
            "use": "Use when exploring how numeric values are distributed.",
        },
        "Box Plot": {
            "explanation": "Summarizes data distribution using median, quartiles, and outliers.",
            "use": "Use to compare spread and skewness across categories.",
        },
        "Violin Plot": {
            "explanation": "Combines a box plot and density plot to show distribution shape and spread.",
            "use": "Use when comparing data distributions across categories.",
        },
        "Heatmap": {
            "explanation": "Uses color to represent numeric relationships or intensities in a grid.",
            "use": "Use for showing correlations or frequency patterns.",
        },
        "Pair Plot": {
            "explanation": "Plots all numeric variables against each other to show relationships.",
            "use": "Use for initial data exploration in multi-variable datasets.",
        },
        "Time Series": {
            "explanation": "Displays how a numeric variable changes over time.",
            "use": "Use for trends, growth, or forecasting analysis.",
        },
        "Bubble Chart": {
            "explanation": "A scatter plot where bubble size shows an extra variable.",
            "use": "Use to add a third dimension like population or magnitude.",
        },
        "Radar Chart": {
            "explanation": "Displays multiple metrics as axes radiating from a center.",
            "use": "Use for skill comparisons or performance profiling.",
        },
        "Polar Chart": {
            "explanation": "Plots data on circular coordinates — useful for cycles.",
            "use": "Use for wind directions, periodic signals, or rotations.",
        },
        "Gauge Chart": {
            "explanation": "Like a speedometer, shows progress or performance toward a goal.",
            "use": "Use for dashboards showing single KPI progress.",
        },
        "Waterfall Chart": {
            "explanation": "Shows cumulative effect of sequential positive/negative values.",
            "use": "Use for profit/loss or budget composition analysis.",
        },
        "Gantt Chart": {
            "explanation": "Visual timeline of tasks with start and end dates.",
            "use": "Use for project scheduling and tracking task overlaps.",
        },
        "Sankey Diagram": {
            "explanation": "Shows flow and volume between connected stages or categories.",
            "use": "Use for process flows, energy transfers, or resource allocation.",
        },
        "Network Graph": {
            "explanation": "Represents connections (edges) between nodes (entities).",
            "use": "Use for social networks, system links, or relationships.",
        },
        "Candlestick Chart": {
            "explanation": "Visualizes open, close, high, and low values — often for financial data.",
            "use": "Use for stock or cryptocurrency price analysis.",
        },
        "Treemap": {
            "explanation": "Shows hierarchical data as nested rectangles sized by value.",
            "use": "Use for displaying part-to-whole relationships in hierarchy.",
        },
        "Sunburst Chart": {
            "explanation": "Displays hierarchy through concentric rings.",
            "use": "Use for multi-level categories or hierarchical breakdowns.",
        },
        "Word Cloud": {
            "explanation": "Highlights most frequent words — bigger words mean higher frequency.",
            "use": "Use for summarizing textual data quickly.",
        },
        "3D Surface Plot": {
            "explanation": "Represents a 3D surface showing peaks and valleys across two variables.",
            "use": "Use for elevation maps, optimization surfaces, or gradients.",
        },
        "3D Scatter Plot": {
            "explanation": "Adds a Z-axis to scatter plots for multi-dimensional insight.",
            "use": "Use when visualizing three numeric dimensions together.",
        },
        "Contour 3D": {
            "explanation": "Displays contour lines over a 3D surface for gradient visualization.",
            "use": "Use when studying density or terrain-like data.",
        },
        "Funnel Chart": {
            "explanation": "Shows decreasing stages in a process, like conversions or drop-offs.",
            "use": "Use for sales pipelines, recruitment, or filtering funnels.",
        },
        "Stacked Area Chart": {
            "explanation": "Displays multiple categories stacked as filled areas over time.",
            "use": "Use to show total and category-specific trends.",
        },
        "Lollipop Chart": {
            "explanation": "A cleaner bar chart alternative — circles on lines.",
            "use": "Use for ranking or comparing categories with fewer distractions.",
        },
        "Hexbin Plot": {
            "explanation": "Groups scatter points into hexagonal bins to show density.",
            "use": "Use for large datasets where points overlap heavily.",
        },
        "Density Contour": {
            "explanation": "Contours show where data points are most concentrated.",
            "use": "Use for visualizing density in scatter-like datasets.",
        },
        "ECDF Plot": {
            "explanation": "Cumulative percentage of observations below each value.",
            "use": "Use for comparing distributions or detecting outliers.",
        },
        "Lag Plot": {
            "explanation": "Compares a time series with its lagged version.",
            "use": "Use to detect autocorrelation patterns in time series data.",
        },
        "Autocorrelation Plot": {
            "explanation": "Shows how values in a time series relate to their past values.",
            "use": "Use for identifying repeating patterns or seasonality.",
        },
        "Parallel Coordinates": {
            "explanation": "Plots multi-variable data as lines across parallel axes.",
            "use": "Use for visualizing many attributes of observations simultaneously.",
        },
        "Violin-Strip Hybrid": {
            "explanation": "Shows both density and individual data points.",
            "use": "Use for a complete view of data distribution and variation.",
        },
        "Ridgeline Plot": {
            "explanation": "Stacks multiple density curves for different categories.",
            "use": "Use when comparing distribution shapes across groups.",
        },
        "Calendar Heatmap": {
            "explanation": "Shows data across days or months in a calendar layout.",
            "use": "Use for visualizing daily activity or trends over time.",
        },
        "Map Scatter Plot": {
            "explanation": "Plots data points as bubbles using latitude and longitude.",
            "use": "Use for showing geographic locations of data.",
        },
        "Choropleth Map": {
            "explanation": "Fills map areas with color intensity based on numeric value.",
            "use": "Use for regional or country-level metrics.",
        },
        "Bubble Timeline": {
            "explanation": "Bubbles along a time axis — size shows an extra variable.",
            "use": "Use for visualizing change in multiple dimensions over time.",
        },
    }

    # --- Summarize dataset briefly ---
    df_summary = ""
    if df is not None:
        try:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            text_cols = df.select_dtypes(exclude='number').columns.tolist()
            df_summary = f"Numeric columns: {numeric_cols}. Text columns: {text_cols}. Rows: {len(df)}."
        except Exception:
            df_summary = "Dataset contains mixed column types."

    # --- Try Groq API ---
    try:
        response = explain_chart_with_llm(plot_type, df_summary)
        if response and "⚠️" not in response:
            return {"explanation": response, "use": f"Use this {plot_type.lower()} to visualize {', '.join(numeric_cols or ['data'])}."}
    except Exception:
        pass

    # --- Fallback static explanation ---
    return info.get(plot_type, {
        "explanation": f"A {plot_type} helps visualize relationships or trends between variables.",
        "use": f"Use the {plot_type} when exploring data patterns."
    })
def explain_concept(term: str):
    """Simple educational glossary."""
    concepts = {
        "correlation": "Correlation measures how two variables move together. A high positive value means they rise together.",
        "outlier": "An outlier is a data point that differs greatly from other observations.",
        "variance": "Variance measures how spread out your data is from the mean.",
        "mean": "Mean is the average value of a numeric column.",
        "median": "Median is the middle value when all data points are ordered.",
        "skew": "Skewness shows if data leans more to one side of the mean.",
    }
    for key in concepts:
        if key in term.lower():
            return concepts[key]
    return "Sorry, I don’t have a definition for that yet."

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
