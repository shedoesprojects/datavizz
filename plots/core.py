"""Core plot implementations with full edge-case handling."""

from typing import Tuple, Any, List, Optional
from io import BytesIO
from .registry import register_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

def _apply_theme(engine, theme_cfg):
    """
    Apply dynamic theme + user color customizations for Matplotlib and Plotly.
    """
    import matplotlib.pyplot as plt
    import plotly.io as pio

    if not theme_cfg:
        return

    theme = theme_cfg.get("theme")
    palette = theme_cfg.get("palette")
    custom_colors = theme_cfg.get("custom_colors", {})

    # --- Matplotlib themes ---
    if engine == "matplotlib":
        if theme == "Seaborn":
            plt.style.use("seaborn-v0_8")
        elif theme == "Plotly Dark":
            plt.style.use("dark_background")
        elif theme == "Minimal":
            plt.style.use("ggplot")
        elif theme == "Custom":
            plt.rcParams.update({
                "axes.facecolor": custom_colors.get("background", "#FFFFFF"),
                "figure.facecolor": custom_colors.get("background", "#FFFFFF"),
                "axes.edgecolor": custom_colors.get("text", "#000000"),
                "text.color": custom_colors.get("text", "#000000"),
                "axes.labelcolor": custom_colors.get("text", "#000000"),
                "xtick.color": custom_colors.get("text", "#000000"),
                "ytick.color": custom_colors.get("text", "#000000"),
            })
            # Set plot elements (bars/lines/points)
            plt.rcParams["axes.prop_cycle"] = plt.cycler(
                color=[custom_colors.get("plot", "#1f77b4")]
            )

        # Handle standard palettes
        if palette and palette != "Default" and theme != "Custom":
            try:
                plt.rcParams["axes.prop_cycle"] = plt.cycler(
                    color=plt.cm.get_cmap(palette).colors
                )
            except Exception:
                pass

    # --- Plotly themes ---
    elif engine == "plotly":
        if theme == "Plotly Dark":
            pio.templates.default = "plotly_dark"
        elif theme == "Seaborn":
            pio.templates.default = "seaborn"
        elif theme == "Minimal":
            pio.templates.default = "plotly_white"
        elif theme == "Custom":
            pio.templates.default = None
            pio.templates["custom"] = pio.templates["plotly"]
            layout = pio.templates["custom"].layout
            layout.paper_bgcolor = custom_colors.get("background", "#FFFFFF")
            layout.plot_bgcolor = custom_colors.get("background", "#FFFFFF")
            layout.font.color = custom_colors.get("text", "#000000")
            layout.colorway = [custom_colors.get("plot", "#1f77b4")]
            pio.templates.default = "custom"


# ----------------- UTILITIES -----------------

def _fig_to_png_bytes(fig) -> bytes:
    """Convert a Matplotlib figure to PNG bytes."""
    from matplotlib import pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    buf.close()
    plt.close(fig)
    return data

def _auto_select_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """
    Return a validated numeric column name.
    If 'preferred' is provided and valid, use it.
    Otherwise, auto-select the first numeric column.

    Raises:
        ValueError: if no numeric columns found or preferred invalid.
    """
    df = _auto_convert_numeric(df)

    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Column '{preferred}' not found in DataFrame.")
        _require_numeric(df, [preferred])
        return preferred

    numeric_cols = df.select_dtypes(include="number").columns
    if numeric_cols.empty:
        raise ValueError("No numeric columns found for auto-selection.")
    return numeric_cols[0]

def _require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

def _require_numeric(df, cols):
    non_num = [c for c in cols if not np.issubdtype(df[c].dtype, np.number)]
    if non_num:
        raise ValueError(f"Non-numeric columns: {', '.join(non_num)}")

def _require_min_numeric(df: pd.DataFrame, min_cols: int):
    """Ensure the DataFrame has at least min_cols numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < min_cols:
        raise ValueError(f"Need at least {min_cols} numeric columns.")

def safe_plot(func):
    """
    Decorator for safe plot execution.
    Catches errors and raises clean, uniform messages.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as ve:
            # validation errors -> show clearly
            raise ValueError(f"Validation Error in {func.__name__}: {ve}")
        except KeyError as ke:
            raise KeyError(f"Missing column in {func.__name__}: {ke}")
        except Exception as e:
            # fallback for any other error
            raise RuntimeError(f"Plot failed in {func.__name__}: {e}")
    return wrapper

def _coerce_datetime(series: pd.Series, col_name: str) -> pd.Series:
    """Try to convert a Series to datetime."""
    s = pd.to_datetime(series, errors="coerce")
    if s.notna().sum() == 0:
        raise ValueError(f"Column '{col_name}' could not be parsed as dates.")
    return s

def _auto_convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detect and convert numeric-looking string columns
    (like '1,200' or '3,456.78') to proper numeric dtype.
    """
    for col in df.columns:
        if df[col].dtype == object:
            # Try removing commas and converting to float
            try:
                cleaned = df[col].str.replace(",", "", regex=False)
                numeric = pd.to_numeric(cleaned, errors="ignore")
                if pd.api.types.is_numeric_dtype(numeric):
                    df[col] = numeric
            except Exception:
                pass  # leave as-is if it fails
    return df

# ----------------- CORE PLOTS (already had) -----------------

@register_plot("Line Plot", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def line_plot(df, x, y, title="", engine="matplotlib", **kwargs) -> Tuple[str, Any]:
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)

    if data.empty:
        raise ValueError("No data to plot after removing NaNs.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.line(data, x=x, y=y, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 4)))
    ax.plot(data[x], data[y], marker=kwargs.get("marker", "o"))
    ax.set_title(title or f"{y} vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True)
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Scatter Plot", params=["x", "y", "hue"], engines=["matplotlib", "plotly"])
@safe_plot
def scatter_plot(df, x, y, hue=None, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    data = df.dropna(subset=[x, y])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data to plot after removing NaNs.")
    if hue and hue not in df.columns:
        raise ValueError(f"Hue column '{hue}' not found.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.scatter(data, x=x, y=y, color=hue, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 6)))
    if hue:
        for name, group in data.groupby(hue):
            ax.scatter(group[x], group[y], label=str(name), alpha=0.8)
        ax.legend()
    else:
        ax.scatter(data[x], data[y], alpha=0.7)
    ax.set_title(title or f"{y} vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Bar Chart", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def bar_chart(df, x, y, title="", engine="matplotlib", horizontal=False, **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data to plot after removing NaNs.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.bar(data, x=y if horizontal else x, y=x if horizontal else y,
                     orientation="h" if horizontal else "v", title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 4)))
    if horizontal:
        ax.barh(data[x].astype(str), data[y])
        ax.set_xlabel(y)
        ax.set_ylabel(x)
    else:
        ax.bar(data[x].astype(str), data[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title or f"{y} by {x}")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Histogram", params=["x"], engines=["matplotlib", "plotly"])
@safe_plot
def histogram_plot(df, x, bins=30, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x])
    _require_numeric(df, [x])
    series = df[x].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if series.empty:
        raise ValueError("No data to plot after removing NaNs.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.histogram(df, x=x, nbins=bins, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 4)))
    ax.hist(series, bins=bins)
    ax.set_title(title or f"Distribution of {x}")
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Box Plot", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def box_plot(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data to plot after removing NaNs.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.box(data, x=x, y=y, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 4)))
    sns.boxplot(data=data, x=x, y=y, ax=ax)
    ax.set_title(title or f"Boxplot of {y} by {x}")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Violin Plot", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def violin_plot(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data to plot after removing NaNs.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.violin(data, x=x, y=y, box=True, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 4)))
    sns.violinplot(data=data, x=x, y=y, ax=ax)
    ax.set_title(title or f"Violin of {y} by {x}")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Heatmap (Correlation)", params=[], engines=["matplotlib"])
@safe_plot
def heatmap_plot(df, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    numeric = df.select_dtypes(include=[np.number])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if numeric.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for a correlation heatmap.")

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))
    sns.heatmap(numeric.corr(), annot=True, cmap="vlag", ax=ax)
    ax.set_title(title or "Correlation Heatmap")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Pair Plot", params=[], engines=["matplotlib"])
@safe_plot
def pair_plot(df, vars: Optional[list] = None, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    numeric = df.select_dtypes(include=["number"])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if vars:
        _require_columns(df, vars)
        numeric = df[vars]
    if numeric.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for a pair plot.")

    import seaborn as sns
    g = sns.pairplot(numeric, corner=True)
    g.fig.suptitle(title or "Pair Plot", y=1.02)
    return "matplotlib", _fig_to_png_bytes(g.fig)


@register_plot("Time Series", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def time_series_plot(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    series = df[[x, y]].dropna()
    series[x] = _coerce_datetime(series[x], x)
    series = series.sort_values(by=x)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.express as px
        fig = px.line(series, x=x, y=y, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 4)))
    ax.plot(series[x], series[y], marker="o")
    ax.set_title(title or f"{y} over {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True)
    return "matplotlib", _fig_to_png_bytes(fig)


# ----------------- BATCH 1 NEW PLOTS -----------------

@register_plot("Stacked Bar Chart", params=["x", "y", "hue"], engines=["matplotlib", "plotly"])
@safe_plot
def stacked_bar(df, x, y, hue=None, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if hue and hue not in df.columns:
        raise ValueError(f"Hue column '{hue}' not found.")
    data = df.dropna(subset=[x, y])

    if engine == "plotly":
        import plotly.express as px
        fig = px.bar(data, x=x, y=y, color=hue, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, errorbar=None)
    ax.set_title(title or f"Stacked Bar of {y} by {x}")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Area Chart", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def area_chart(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.express as px
        fig = px.area(data, x=x, y=y, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 4)))
    ax.fill_between(data[x], data[y], alpha=0.5)
    ax.plot(data[x], data[y], marker="o", color="blue")
    ax.set_title(title or f"Area of {y} over {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Pie Chart", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def pie_chart(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data available for pie chart.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.pie(data, names=x, values=y, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 6)))
    ax.pie(data[y], labels=data[x].astype(str), autopct=lambda p: '{:.0f}'.format(p * sum(data[y]) / 100), startangle=90)
    ax.set_title(title or f"Distribution of {y} by {x}")
    return "matplotlib", _fig_to_png_bytes(fig)

# ----------------- BATCH 2 NEW PLOTS -----------------

@register_plot("Donut Chart", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def donut_chart(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data available for donut chart.")
    
    #if engine == "plotly":
    #    fig = px.pie(data, names=x, values=y, hole=0.4, title=title)
    #    fig.update_traces(textinfo="value+label")  # show raw numbers + labels
    #    return "plotly", fig

    if engine == "plotly":
        import plotly.express as px
        fig = px.pie(data, names=x, values=y, hole=0.4, title=title)
        fig.update_traces(textinfo="value")  # show raw numbers + labelss
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 6)))
    wedges, texts, autotexts = ax.pie(
        data[y], labels=data[x].astype(str), autopct=lambda p: '{:.0f}'.format(p * sum(data[y]) / 100),
        startangle=90, wedgeprops=dict(width=0.4)
    )
    ax.set_title(title or f"Donut chart of {y} by {x}")
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Density Contour", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def density_contour(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [x, y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for density contour.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.density_contour(data, x=x, y=y, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(data=data, x=x, y=y, fill=False, levels=10, ax=ax)
    ax.set_title(title or f"Density Contour of {y} vs {x}")
    return "matplotlib", _fig_to_png_bytes(fig)

# ----------------- BATCH 3 NEW PLOTS -----------------

@register_plot("Word Cloud", params=["text"], engines=["matplotlib"])
@safe_plot
def wordcloud_plot(df, text, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [text])
    series = df[text].dropna().astype(str)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if series.empty:
        raise ValueError("No text data for word cloud.")

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    text_data = " ".join(series)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title or "Word Cloud")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Sankey Diagram", params=["source", "target", "value"], engines=["plotly"])
@safe_plot
def sankey_chart(df, source, target, value, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [source, target, value])
    _require_numeric(df, [value])
    data = df[[source, target, value]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for Sankey diagram.")

    import plotly.graph_objects as go

    sources = data[source].astype(str).tolist()
    targets = data[target].astype(str).tolist()
    values = data[value].tolist()

    labels = list(set(sources + targets))
    label_to_index = {label: i for i, label in enumerate(labels)}

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=20, thickness=20),
        link=dict(
            source=[label_to_index[s] for s in sources],
            target=[label_to_index[t] for t in targets],
            value=values,
        )
    ))
    fig.update_layout(title_text=title or "Sankey Diagram")
    return "plotly", fig

@register_plot("Parallel Coordinates", params=["dimension_columns"], engines=["plotly"])
@safe_plot
def parallel_coordinates(df, dimension_columns: List[str], title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, dimension_columns)
    _require_numeric(df, dimension_columns)
    data = df[dimension_columns].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for parallel coordinates.")

    import plotly.express as px
    fig = px.parallel_coordinates(data, dimensions=dimension_columns, title=title)
    return "plotly", fig

@register_plot("Strip Plot", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def strip_plot(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for strip plot.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.strip(data, x=x, y=y, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.stripplot(data=data, x=x, y=y, jitter=True, ax=ax)
    ax.set_title(title or f"Strip Plot of {y} by {x}")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Step Plot", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def step_plot(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for step plot.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.line(data, x=x, y=y, title=title, line_shape="hv")
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(data[x], data[y], where="mid")
    ax.set_title(title or f"Step Plot of {y} vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Gantt Chart", params=["task", "start", "finish"], engines=["plotly"])
@safe_plot
def gantt_chart(df, task, start, finish, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [task, start, finish])
    data = df[[task, start, finish]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for Gantt chart.")

    data[start] = _coerce_datetime(data[start], start)
    data[finish] = _coerce_datetime(data[finish], finish)

    import plotly.express as px
    fig = px.timeline(data, x_start=start, x_end=finish, y=task, title=title)
    fig.update_yaxes(autorange="reversed")
    return "plotly", fig

@register_plot("Waterfall Chart", params=["x", "y"], engines=["plotly"])
@safe_plot
def waterfall_chart(df, x, y, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for waterfall chart.")

    import plotly.graph_objects as go
    measure = ["relative"] * len(data)
    measure[0] = "absolute"

    fig = go.Figure(go.Waterfall(
        name="Waterfall",
        orientation="v",
        x=data[x].astype(str),
        y=data[y],
        measure=measure
    ))
    fig.update_layout(title=title or f"Waterfall of {y} by {x}")
    return "plotly", fig

@register_plot("Funnel Chart", params=["x", "y"], engines=["plotly"])
@safe_plot
def funnel_chart(df, x, y, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for funnel chart.")

    import plotly.express as px
    fig = px.funnel(data, x=y, y=x, title=title)
    return "plotly", fig

# ----------------- BATCH 4 NEW PLOTS -----------------

@register_plot("Choropleth Map", params=["location", "value"], engines=["plotly"])
@safe_plot
def choropleth_map(df, location, value, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [location, value])
    _require_numeric(df, [value])
    data = df[[location, value]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for choropleth map.")

    import plotly.express as px
    fig = px.choropleth(
        data,
        locations=location,
        locationmode="country names",
        color=value,
        title=title
    )
    return "plotly", fig

@register_plot("Scatter Geo Map", params=["lat", "lon", "size"], engines=["plotly"])
@safe_plot
def scatter_geo_map(df, lat, lon, size=None, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [lat, lon])
    data = df.dropna(subset=[lat, lon])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for scatter geo map.")

    import plotly.express as px
    fig = px.scatter_geo(
        data,
        lat=lat,
        lon=lon,
        size=size if size in df.columns else None,
        title=title
    )
    return "plotly", fig

@register_plot("Candlestick Chart", params=["date", "open", "high", "low", "close"], engines=["plotly"])
@safe_plot
def candlestick_chart(df, date, open, high, low, close, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [date, open, high, low, close])
    data = df[[date, open, high, low, close]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for candlestick chart.")

    data[date] = _coerce_datetime(data[date], date)

    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(
        x=data[date],
        open=data[open],
        high=data[high],
        low=data[low],
        close=data[close]
    )])
    fig.update_layout(title=title or "Candlestick Chart", xaxis_title=date, yaxis_title="Price")
    return "plotly", fig

@register_plot("Treemap", params=["path", "value"], engines=["plotly"])
@safe_plot
def treemap_chart(df, path, value, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [path, value])
    _require_numeric(df, [value])
    data = df[[path, value]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for treemap chart.")

    import plotly.express as px
    fig = px.treemap(data, path=[path], values=value, title=title)
    return "plotly", fig

@register_plot("Sunburst Chart", params=["path", "value"], engines=["plotly"])
@safe_plot
def sunburst_chart(df, path, value, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [path, value])
    _require_numeric(df, [value])
    data = df[[path, value]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for sunburst chart.")

    import plotly.express as px
    fig = px.sunburst(data, path=[path], values=value, title=title)
    return "plotly", fig

@register_plot("Icicle Chart", params=["path", "value"], engines=["plotly"])
@safe_plot
def icicle_chart(df, path, value, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [path, value])
    _require_numeric(df, [value])
    data = df[[path, value]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for icicle chart.")

    import plotly.express as px
    fig = px.icicle(data, path=[path], values=value, title=title)
    return "plotly", fig

# ----------------- BATCH 5 NEW PLOTS -----------------

@register_plot("Bubble Chart", params=["x", "y", "size", "hue"], engines=["plotly"])
@safe_plot
def bubble_chart(df, x, y, size, hue=None, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y, size])
    _require_numeric(df, [x, y, size])
    data = df[[x, y, size] + ([hue] if hue else [])].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for bubble chart.")

    import plotly.express as px
    fig = px.scatter(data, x=x, y=y, size=size, color=hue, title=title, hover_name=x)
    return "plotly", fig

@register_plot("Hexbin Plot", params=["x", "y"], engines=["matplotlib"])
@safe_plot
def hexbin_plot(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [x, y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for hexbin plot.")

    import matplotlib.pyplot as plt
    plt.close("all")
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 5)))
    hb = ax.hexbin(data[x], data[y], gridsize=30, cmap="viridis")
    fig.colorbar(hb, ax=ax)
    ax.set_title(title or f"Hexbin of {y} vs {x}")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Dendrogram", params=["columns"], engines=["matplotlib"])
@safe_plot
def dendrogram_plot(df, columns, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, columns)
    _require_numeric(df, columns)
    data = df[columns].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for dendrogram.")

    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    plt.close("all")
    linked = linkage(data, "ward")
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))
    dendrogram(linked, labels=data.index.astype(str).tolist(), ax=ax)
    ax.set_title(title or "Dendrogram (Hierarchical Clustering)")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Streamgraph", params=["x", "y", "hue"], engines=["plotly"])
@safe_plot
def streamgraph(df, x, y, hue, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y, hue])
    _require_numeric(df, [y])
    data = df[[x, y, hue]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for streamgraph.")

    import plotly.express as px
    fig = px.area(data, x=x, y=y, color=hue, line_group=hue, title=title, groupnorm="fraction")
    return "plotly", fig

@register_plot("Gauge Chart", params=["value"], engines=["plotly"])
@safe_plot
def gauge_chart(df, value, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [value])
    _require_numeric(df, [value])
    val = df[value].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if val.empty:
        raise ValueError("No numeric values found for gauge chart.")

    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(val.mean()),
        gauge={"axis": {"range": [None, val.max()]}},
        title={"text": title or f"Gauge of {value}"}
    ))
    return "plotly", fig

@register_plot("Joy/Ridge Plot", params=["x", "hue"], engines=["matplotlib"])
@safe_plot
def ridge_plot(df, x, hue, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, hue])
    _require_numeric(df, [x])
    data = df[[x, hue]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data for ridge plot.")

    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.close("all")
    g = sns.FacetGrid(data, row=hue, hue=hue, aspect=4, height=1.5, palette="tab10")
    g.map(sns.kdeplot, x, fill=True)
    g.map(plt.axhline, y=0, lw=2)
    g.fig.suptitle(title or f"Ridge Plot of {x} by {hue}", y=1.02)
    return "matplotlib", _fig_to_png_bytes(g.fig)


@register_plot("Network Graph", params=["source", "target"], engines=["plotly"])
@safe_plot
def network_graph(df, source, target, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [source, target])
    edges = df[[source, target]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if edges.empty:
        raise ValueError("No data for network graph.")

    import networkx as nx
    import plotly.graph_objects as go
    G = nx.from_pandas_edgelist(edges, source=source, target=target)

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#888"))
    node_x, node_y = zip(*pos.values())
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=list(G.nodes),
        marker=dict(size=10, color="blue"), textposition="top center"
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(title=title or "Network Graph", showlegend=False)
    return "plotly", fig

# ---------------- BATCH 6 NEW PLOTS---------------- #

@register_plot("Population Pyramid", params=["x", "y", "hue"], engines=["matplotlib", "plotly"])
@safe_plot
def population_pyramid(df, x, y, hue=None, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df.dropna(subset=[x, y])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if hue and hue not in df.columns:
        raise ValueError(f"Hue column '{hue}' not found.")

    if engine == "plotly":
        import plotly.express as px
        fig = px.bar(data, x=y, y=x, color=hue, orientation="h", barmode="relative", title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))
    if hue:
        categories = data[hue].unique()
        for cat in categories:
            subset = data[data[hue] == cat]
            ax.barh(subset[x], subset[y] * (1 if cat == categories[0] else -1), label=str(cat))
        ax.legend()
    else:
        ax.barh(data[x], data[y])
    ax.set_title(title or "Population Pyramid")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Stacked Bar Multi", params=["x", "y_cols"], engines=["matplotlib", "plotly"])
@safe_plot
def stacked_bar_multi(df, x, y_cols: list, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x] + y_cols)
    _require_numeric(df, y_cols)
    data = df[[x] + y_cols].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data to plot.")

    # reshape to long format for consistency
    melted = data.melt(id_vars=x, value_vars=y_cols, var_name="Variable", value_name="Value")

    if engine == "plotly":
        import plotly.express as px
        fig = px.bar(melted, x=x, y="Value", color="Variable", title=title, barmode="stack")
        return "plotly", fig

    import matplotlib.pyplot as plt
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))
    melted_pivot = data.set_index(x)
    melted_pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title or "Stacked Bar Chart")
    return "matplotlib", _fig_to_png_bytes(fig)



@register_plot("Grouped Bar Chart", params=["x", "y_cols"], engines=["matplotlib", "plotly"])
@safe_plot
def grouped_bar_chart(df, x, y_cols, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    """
    Grouped bar chart: multiple y columns compared side by side for each x.
    Example: x=Year, y_cols=["Applied", "Shortlisted", "Qualified"]
    """
    _require_columns(df, [x] + y_cols)
    _require_numeric(df, y_cols)

    data = df[[x] + y_cols].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if data.empty:
        raise ValueError("No data available for grouped bar chart.")

    if engine == "plotly":
        import plotly.express as px
        melted = data.melt(id_vars=x, value_vars=y_cols, var_name="Variable", value_name="Value")
        fig = px.bar(melted, x=x, y="Value", color="Variable", barmode="group", title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    import numpy as np
    plt.close("all")

    n_y = len(y_cols)
    x_vals = np.arange(len(data[x]))
    bar_width = 0.8 / n_y

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
    for i, col in enumerate(y_cols):
        ax.bar(x_vals + i * bar_width, data[col], width=bar_width, label=col)

    ax.set_xticks(x_vals + bar_width * (n_y - 1) / 2)
    ax.set_xticklabels(data[x].astype(str))
    ax.set_xlabel(x)
    ax.set_ylabel("Value")
    ax.set_title(title or f"Grouped Bar Chart of {', '.join(y_cols)} by {x}")
    ax.legend()
    return "matplotlib", _fig_to_png_bytes(fig)



@register_plot("Stacked Histogram", params=["x", "hue"], engines=["matplotlib", "plotly"])
@safe_plot
def stacked_histogram(df, x, hue=None, bins=30, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x])
    _require_numeric(df, [x])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.express as px
        fig = px.histogram(df, x=x, color=hue, nbins=bins, barmode="stack", title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))
    if hue:
        groups = [df[df[hue] == cat][x].dropna() for cat in df[hue].dropna().unique()]
        ax.hist(groups, bins=bins, stacked=True, label=df[hue].unique())
        ax.legend()
    else:
        ax.hist(df[x].dropna(), bins=bins)
    ax.set_title(title or f"Stacked Histogram of {x}")
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Swarm Plot", params=["x", "y"], engines=["matplotlib"])
@safe_plot
def swarm_plot(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.swarmplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title or "Swarm Plot")
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Cumulative Sum Plot", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def cumsum_plot(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    data["cumsum"] = data[y].cumsum()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.express as px
        fig = px.line(data, x=x, y="cumsum", title=title or f"Cumulative Sum of {y}")
        return "plotly", fig

    import matplotlib.pyplot as plt
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data[x], data["cumsum"], marker="o")
    ax.set_title(title or f"Cumulative Sum of {y}")
    return "matplotlib", _fig_to_png_bytes(fig)


# ----------------- BATCH 7 NEW PLOTS updated-----------------

@register_plot('QQ Plot', params=["column"], engines=["matplotlib"])
@safe_plot
def qq_plot(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Quantile-Quantile Plot to compare data distribution against a normal distribution.
    
    Args:
        df: pandas DataFrame
        column: Column name to plot (if None, uses first numeric column)
        title: Plot title
        engine: Only 'matplotlib' supported
        **kwargs: Additional arguments (figsize, etc.)
    
    Returns:
        Tuple[str, bytes]: ('matplotlib', png_bytes)
    """
    df = _auto_convert_numeric(df)
    column = _auto_select_column(df, column)

    import scipy.stats as stats
    import matplotlib.pyplot as plt
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 6)))
    stats.probplot(df[column], dist="norm", plot=ax)
    ax.set_title(title or f"QQ Plot of {column}")
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('ECDF Plot', params=["column"], engines=["matplotlib"])
@safe_plot
def ecdf_plot(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Empirical Cumulative Distribution Function plot.
    
    Args:
        df: pandas DataFrame
        column: Column name to plot (if None, uses first numeric column)
        title: Plot title
        engine: Only 'matplotlib' supported
        **kwargs: Additional arguments (figsize, etc.)
    
    Returns:
        Tuple[str, bytes]: ('matplotlib', png_bytes)
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    # Auto-select column if not provided
    column = _auto_select_column(df, column)
    
    import matplotlib.pyplot as plt
    
    x = np.sort(df[column].dropna())
    y = np.arange(1, len(x)+1) / len(x)
    
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 4)))
    ax.plot(x, y, marker='.', linestyle='none')
    ax.set_xlabel(column)
    ax.set_ylabel("ECDF")
    ax.set_title(title or f"ECDF Plot of {column}")
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Lag Plot', params=["column"], engines=["matplotlib"])
@safe_plot
def lag_plot(df, column=None, lag=1, title="", engine="matplotlib", **kwargs):
    """
    Lag Plot helps identify autocorrelation within time series data.
    
    Args:
        df: pandas DataFrame
        column: Column name to plot (if None, uses first numeric column)
        lag: Number of lags (default: 1)
        title: Plot title
        engine: Only 'matplotlib' supported
        **kwargs: Additional arguments (figsize, etc.)
    
    Returns:
        Tuple[str, bytes]: ('matplotlib', png_bytes)
    """
    df = _auto_convert_numeric(df)
    from pandas.plotting import lag_plot as pd_lag_plot
    import matplotlib.pyplot as plt
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    # Auto-select column if not provided
    column = _auto_select_column(df, column)

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 4)))
    pd_lag_plot(df[column], lag=lag, ax=ax)
    ax.set_title(title or f"Lag Plot of {column}")
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Autocorrelation Plot', params=["column"], engines=["matplotlib"])
@safe_plot
def autocorr_plot(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Displays correlation of a variable with itself over successive time lags.
    """
    df = _auto_convert_numeric(df)
    from pandas.plotting import autocorrelation_plot
    import matplotlib.pyplot as plt
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    column = _auto_select_column(df, column)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 4))
    autocorrelation_plot(df[column], ax=ax)
    ax.set_title(title or f"Autocorrelation Plot of {column}")
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('KDE Comparison Plot', params=["columns"], engines=["matplotlib"])
@safe_plot
def kde_comparison(df, columns=None, title="", engine="matplotlib", **kwargs):
    """
    Compare Kernel Density Estimation of multiple numeric columns.
    """
    df = _auto_convert_numeric(df)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if not numeric_cols:
        raise ValueError("No numeric columns found for KDE Comparison Plot.")
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 5))
    for col in numeric_cols:
        sns.kdeplot(data=df, x=col, fill=True, label=col, ax=ax)
    ax.set_title(title or "KDE Comparison Plot")
    ax.legend()
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Mean-Variance Plot', params=[], engines=["matplotlib"])
@safe_plot
def mean_variance_plot(df, title="", engine="matplotlib", **kwargs):
    """
    Displays the mean and variance of numeric columns for quick comparison.
    """
    df = _auto_convert_numeric(df)
    numeric_cols = df.select_dtypes(include='number').columns
    if numeric_cols.empty:
        raise ValueError("No numeric columns found for Mean-Variance Plot.")
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    mean_vals = df[numeric_cols].mean()
    var_vals = df[numeric_cols].var()
    
    plt.figure(figsize=(7, 5))
    plt.scatter(mean_vals, var_vals)
    for i, col in enumerate(numeric_cols):
        plt.text(mean_vals[i], var_vals[i], col)
    plt.xlabel("Mean")
    plt.ylabel("Variance")
    plt.title("Mean vs Variance Plot")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Diverging Bar Plot', params=["column"], engines=["matplotlib"])
@safe_plot
def diverging_bar(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Highlights deviation of values from mean using diverging bars.
    """
    df = _auto_convert_numeric(df)
    numeric_cols = df.select_dtypes(include='number').columns
    if numeric_cols.empty:
        raise ValueError("No numeric columns found for Diverging Bar Plot.")
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    col = numeric_cols[0]
    mean_val = df[col].mean()
    df_sorted = df.sort_values(by=col)
    colors = df_sorted[col].apply(lambda x: 'green' if x >= mean_val else 'red')
    
    plt.figure(figsize=(8, 5))
    plt.bar(df_sorted.index, df_sorted[col] - mean_val, color=colors)
    plt.axhline(0, color='black', lw=1)
    plt.title(f"Diverging Bar Plot ({col})")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

# ----------------- BATCH 8 NEW PLOTS updated -----------------

@register_plot('OHLC Chart', params=["date", "open", "high", "low", "close"], engines=["plotly"])
@safe_plot
def ohlc_chart(df, date=None, open=None, high=None, low=None, close=None, title="", engine="plotly", **kwargs):
    """
    Open-High-Low-Close chart for stock price movement visualization.
    """
    df = _auto_convert_numeric(df)
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("OHLC Chart requires Open, High, Low, and Close columns.")
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig = go.Figure(data=[go.Ohlc(
        x=df.index if 'Date' not in df.columns else df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close']
    )])
    fig.update_layout(title="OHLC Chart", xaxis_title="Time", yaxis_title="Price")
    return "plotly", fig

@register_plot('Bollinger Bands', params=["close"], engines=["matplotlib"])
@safe_plot
def bollinger_bands(df, close=None, title="", engine="matplotlib", **kwargs):
    """
    Displays Bollinger Bands using closing price with 20-day moving average and ±2 std dev.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Close' not in df.columns:
        raise ValueError("Bollinger Bands requires a 'Close' column.")
    
    window = 20
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['Upper'] = df['MA'] + 2 * df['Close'].rolling(window=window).std()
    df['Lower'] = df['MA'] - 2 * df['Close'].rolling(window=window).std()

    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['MA'], label='Moving Average', color='orange')
    plt.fill_between(df.index, df['Upper'], df['Lower'], color='lightgray', alpha=0.4, label='Bollinger Bands')
    plt.title("Bollinger Bands (20-Day, 2σ)")
    plt.legend()
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Moving Average Plot', params=["close"], engines=["matplotlib"])
@safe_plot
def moving_average_plot(df, close=None, title="", engine="matplotlib", **kwargs):
    """
    Plots Close price with short-term and long-term moving averages.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Close' not in df.columns:
        raise ValueError("Moving Average Plot requires a 'Close' column.")
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['MA20'], label='20-Day MA', color='orange')
    plt.plot(df['MA50'], label='50-Day MA', color='green')
    plt.title("Moving Average Plot")
    plt.legend()
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Volume Chart', params=["volume"], engines=["matplotlib"])
@safe_plot
def volume_chart(df, volume=None, title="", engine="matplotlib", **kwargs):
    """
    Volume chart showing traded volume over time.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Volume' not in df.columns:
        raise ValueError("Volume Chart requires a 'Volume' column.")
    
    plt.figure(figsize=(10, 4))
    plt.bar(df.index if 'Date' not in df.columns else df['Date'], df['Volume'], color='purple')
    plt.title("Volume Chart")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Returns Histogram', params=["close"], engines=["matplotlib"])
@safe_plot
def returns_histogram(df, close=None, title="", engine="matplotlib", **kwargs):
    """
    Histogram of daily percentage returns for stock prices.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Close' not in df.columns:
        raise ValueError("Returns Histogram requires a 'Close' column.")
    
    returns = df['Close'].pct_change().dropna()
    plt.figure(figsize=(8, 4))
    plt.hist(returns, bins=40, color='teal', alpha=0.7)
    plt.title("Histogram of Daily Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

# ----------------- BATCH 9 NEW PLOTS -----------------

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

@register_plot('Confusion Matrix', params=["actual", "predicted"], engines=["matplotlib"])
@safe_plot
def confusion_matrix_plot(df, actual=None, predicted=None, title="", engine="matplotlib", **kwargs):
    """
    Displays Confusion Matrix heatmap.
    Expects 'Actual' and 'Predicted' columns with categorical or binary values.
    """
    if 'Actual' not in df.columns or 'Predicted' not in df.columns:
        raise ValueError("Confusion Matrix requires 'Actual' and 'Predicted' columns.")
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    cm = confusion_matrix(df['Actual'], df['Predicted'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('ROC Curve', params=["actual", "predicted_prob"], engines=["matplotlib"])
@safe_plot
def roc_curve_plot(df, actual=None, predicted_prob=None, title="", engine="matplotlib", **kwargs):
    """
    Plots Receiver Operating Characteristic (ROC) curve.
    Expects 'Actual' (0/1) and 'Predicted_Prob' columns.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Actual' not in df.columns or 'Predicted_Prob' not in df.columns:
        raise ValueError("ROC Curve requires 'Actual' and 'Predicted_Prob' columns.")
    
    fpr, tpr, _ = roc_curve(df['Actual'], df['Predicted_Prob'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Precision-Recall Curve', params=["actual", "predicted_prob"], engines=["matplotlib"])
@safe_plot
def precision_recall_curve_plot(df, actual=None, predicted_prob=None, title="", engine="matplotlib", **kwargs):
    """
    Displays Precision-Recall curve for classification models.
    Expects 'Actual' and 'Predicted_Prob' columns.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Actual' not in df.columns or 'Predicted_Prob' not in df.columns:
        raise ValueError("Precision-Recall Curve requires 'Actual' and 'Predicted_Prob' columns.")
    
    precision, recall, _ = precision_recall_curve(df['Actual'], df['Predicted_Prob'])
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='purple')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Feature Importance', params=["feature", "importance"], engines=["matplotlib"])
@safe_plot
def feature_importance_plot(df, feature=None, importance=None, title="", engine="matplotlib", **kwargs):
    """
    Displays feature importance based on model output.
    Expects columns: 'Feature' and 'Importance'.
    """
    if 'Feature' not in df.columns or 'Importance' not in df.columns:
        raise ValueError("Feature Importance Plot requires 'Feature' and 'Importance' columns.")
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    df_sorted = df.sort_values('Importance', ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(df_sorted['Feature'], df_sorted['Importance'], color='orange')
    plt.title("Feature Importance Plot")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Residual Plot', params=["actual", "predicted"], engines=["matplotlib"])
@safe_plot
def residual_plot(df, actual=None, predicted=None, title="", engine="matplotlib", **kwargs):
    """
    Plots residuals (Actual - Predicted) to visualize error distribution.
    Expects 'Actual' and 'Predicted' columns.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Actual' not in df.columns or 'Predicted' not in df.columns:
        raise ValueError("Residual Plot requires 'Actual' and 'Predicted' columns.")
    
    df['Residual'] = df['Actual'] - df['Predicted']
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df['Predicted'], y=df['Residual'])
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Lift Chart', params=["actual", "predicted_prob"], engines=["matplotlib"])
@safe_plot
def lift_chart(df, actual=None, predicted_prob=None, title="", engine="matplotlib", **kwargs):
    """
    Visualizes model lift across deciles.
    Expects 'Actual' (0/1) and 'Predicted_Prob' columns.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Actual' not in df.columns or 'Predicted_Prob' not in df.columns:
        raise ValueError("Lift Chart requires 'Actual' and 'Predicted_Prob' columns.")
    
    df_sorted = df.sort_values('Predicted_Prob', ascending=False)
    df_sorted['decile'] = pd.qcut(df_sorted['Predicted_Prob'], 10, labels=False)
    lift = df_sorted.groupby('decile')['Actual'].mean() / df_sorted['Actual'].mean()

    plt.figure(figsize=(7, 5))
    plt.plot(lift.index, lift.values, marker='o')
    plt.title("Lift Chart")
    plt.xlabel("Decile")
    plt.ylabel("Lift")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Gain Chart', params=["actual", "predicted_prob"], engines=["matplotlib"])
@safe_plot
def gain_chart(df, actual=None, predicted_prob=None, title="", engine="matplotlib", **kwargs):
    """
    Cumulative Gain Chart showing model effectiveness.
    Expects 'Actual' and 'Predicted_Prob' columns.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Actual' not in df.columns or 'Predicted_Prob' not in df.columns:
        raise ValueError("Gain Chart requires 'Actual' and 'Predicted_Prob' columns.")
    
    df_sorted = df.sort_values('Predicted_Prob', ascending=False)
    df_sorted['cum_actual'] = df_sorted['Actual'].cumsum() / df_sorted['Actual'].sum()
    df_sorted['cum_obs'] = np.arange(1, len(df_sorted)+1) / len(df_sorted)

    plt.figure(figsize=(7, 5))
    plt.plot(df_sorted['cum_obs'], df_sorted['cum_actual'], label='Model', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline')
    plt.title("Cumulative Gain Chart")
    plt.xlabel("Proportion of Observations")
    plt.ylabel("Proportion of Positives Captured")
    plt.legend()
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

# ----------------- BATCH 10 NEW PLOTS -----------------

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

@register_plot('Rolling Mean & Std Plot', params=["column"], engines=["matplotlib"])
@safe_plot
def rolling_stats_plot(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Displays rolling mean and standard deviation for a time series column.
    """
    df = _auto_convert_numeric(df)
    column = _auto_select_column(df, column)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Date' in df.columns:
        df = df.set_index(pd.to_datetime(df['Date']))

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[column], label='Original', color='blue')
    ax.plot(df[column].rolling(12).mean(), label='Rolling Mean', color='orange')
    ax.plot(df[column].rolling(12).std(), label='Rolling Std', color='green')
    ax.set_title(title or f"Rolling Mean & Std of {column}")
    ax.legend()
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Seasonal Decomposition', params=["column"], engines=["matplotlib"])
@safe_plot
def seasonal_decomposition_plot(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Decomposes a time series into Trend, Seasonality, and Residual components.
    """
    df = _auto_convert_numeric(df)
    if 'Date' in df.columns:
        df = df.set_index(pd.to_datetime(df['Date']))
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    column = _auto_select_column(df, column)

    from statsmodels.tsa.seasonal import seasonal_decompose
    import matplotlib.pyplot as plt
    
    plt.close('all')
    decomposition = seasonal_decompose(df[column].dropna(), model='additive', period=12)
    fig = decomposition.plot()
    fig.suptitle(title or f"Seasonal Decomposition of {column}", fontsize=14)
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Forecast Plot (Linear Trend)', params=["column"], engines=["matplotlib"])
@safe_plot
def forecast_plot(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Simple forecast visualization using linear regression for trend projection.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Date' not in df.columns:
        raise ValueError("Forecast Plot requires a 'Date' column.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    column = _auto_select_column(df, column)

    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[column].values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(df) + 12).reshape(-1, 1)
    future_y = model.predict(future_X)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], y, label='Actual', color='blue')
    ax.plot(pd.date_range(df['Date'].iloc[0], periods=len(future_y), freq='M'), 
            future_y, '--', color='orange', label='Forecast')
    ax.set_title(title or f"Forecast Plot (Linear Trend) for {column}")
    ax.legend()
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Trend + Seasonality Plot', params=["column"], engines=["matplotlib"])
@safe_plot
def trend_seasonality_plot(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Combines rolling mean and seasonal decomposition for deeper time series understanding.
    """
    df = _auto_convert_numeric(df)
    if 'Date' in df.columns:
        df = df.set_index(pd.to_datetime(df['Date']))
    
    column = _auto_select_column(df, column)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    from statsmodels.tsa.seasonal import seasonal_decompose
    import matplotlib.pyplot as plt
    
    rolling_mean = df[column].rolling(12).mean()
    decomposition = seasonal_decompose(df[column].dropna(), model='additive', period=12)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df[column], label='Original', alpha=0.6)
    ax.plot(rolling_mean, label='Rolling Mean', color='orange')
    ax.plot(decomposition.trend, label='Trend', color='green')
    ax.plot(decomposition.seasonal, label='Seasonal', color='red', linestyle='--')
    ax.set_title(title or f"Trend + Seasonality of {column}")
    ax.legend()
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Time Series Heatmap', params=["column"], engines=["matplotlib"])
@safe_plot
def time_series_heatmap(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Creates a monthly heatmap of average values over years.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Date' not in df.columns:
        raise ValueError("Time Series Heatmap requires a 'Date' column.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    column = _auto_select_column(df, column)

    plt.close('all')
    pivot = df.pivot_table(values=column, index='Year', columns='Month', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt=".1f", ax=ax)
    ax.set_title(title or f"Time Series Heatmap of {column}")
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Anomaly Detection Plot', params=["column"], engines=["matplotlib"])
@safe_plot
def anomaly_detection_plot(df, column=None, title="", engine="matplotlib", **kwargs):
    """
    Highlights anomalies in a time series using standard deviation thresholds.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if 'Date' not in df.columns:
        raise ValueError("Anomaly Detection Plot requires a 'Date' column.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    column = _auto_select_column(df, column)

    import matplotlib.pyplot as plt
    
    mean, std = df[column].mean(), df[column].std()
    upper, lower = mean + 2 * std, mean - 2 * std
    anomalies = df[(df[column] > upper) | (df[column] < lower)]

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df[column], color='blue', label='Data')
    ax.scatter(anomalies['Date'], anomalies[column], color='red', label='Anomalies', zorder=5)
    ax.axhline(upper, color='orange', linestyle='--', label='Upper Bound')
    ax.axhline(lower, color='orange', linestyle='--', label='Lower Bound')
    ax.set_title(title or f"Anomaly Detection Plot ({column})")
    ax.legend()
    
    return "matplotlib", _fig_to_png_bytes(fig)

# ----------------- BATCH 11 NEW PLOTS -----------------

@register_plot('3D Line Plot', params=["x_col", "y_col", "z_col"], engines=["plotly"])
@safe_plot
def line_3d_plot(df, x_col=None, y_col=None, z_col=None, title="", engine="plotly", **kwargs):
    """
    Creates a 3D line plot using three numeric columns as X, Y, Z.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 3:
        raise ValueError("3D Line Plot requires at least three numeric columns (X, Y, Z).")

    fig = go.Figure(data=[go.Scatter3d(
        x=df[numeric_cols[0]], y=df[numeric_cols[1]], z=df[numeric_cols[2]],
        mode='lines', line=dict(color='blue', width=3)
    )])
    fig.update_layout(scene=dict(
        xaxis_title=numeric_cols[0], yaxis_title=numeric_cols[1], zaxis_title=numeric_cols[2]
    ), title="3D Line Plot")
    return "plotly", fig

@register_plot('3D Surface Plot', params=["x_col", "y_col", "z_col"], engines=["plotly"])
@safe_plot
def surface_plot(df, x_col=None, y_col=None, z_col=None, title="", engine="plotly", **kwargs):
    """
    Creates a 3D surface plot for mesh-like data (Z as function of X, Y).
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if df.shape[1] < 3:
        raise ValueError("3D Surface Plot requires at least 3 numeric columns (X, Y, Z).")

    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    fig = go.Figure(data=[go.Surface(z=z.values.reshape(int(np.sqrt(len(z))), -1))])
    fig.update_layout(title="3D Surface Plot", scene=dict(
        xaxis_title=df.columns[0], yaxis_title=df.columns[1], zaxis_title=df.columns[2]
    ))
    return "plotly", fig

@register_plot('3D Mesh Plot', params=["x_col", "y_col", "z_col"], engines=["plotly"])
@safe_plot
def mesh_3d_plot(df, x_col=None, y_col=None, z_col=None, title="", engine="plotly", **kwargs):
    """
    Creates a 3D mesh representation using Plotly.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 3:
        raise ValueError("3D Mesh Plot requires at least 3 numeric columns (X, Y, Z).")

    fig = go.Figure(data=[go.Mesh3d(
        x=df[numeric_cols[0]], y=df[numeric_cols[1]], z=df[numeric_cols[2]],
        opacity=0.5, color='skyblue'
    )])
    fig.update_layout(title="3D Mesh Plot", scene=dict(
        xaxis_title=numeric_cols[0], yaxis_title=numeric_cols[1], zaxis_title=numeric_cols[2]
    ))
    return "plotly", fig

@register_plot('3D Histogram', params=["x_col", "y_col"], engines=["plotly"])
@safe_plot
def histogram_3d_plot(df, x_col=None, y_col=None, title="", engine="plotly", **kwargs):
    """
    Creates a 3D histogram using X, Y numeric columns.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 2:
        raise ValueError("3D Histogram requires at least two numeric columns.")

    fig = go.Figure(data=[go.Histogram2dContour(
        x=df[numeric_cols[0]], y=df[numeric_cols[1]], contours=dict(coloring='heatmap')
    )])
    fig.update_layout(title="3D Histogram (2D Density)", xaxis_title=numeric_cols[0], yaxis_title=numeric_cols[1])
    return "plotly", fig

@register_plot('3D Contour Plot', params=["x_col", "y_col", "z_col"], engines=["plotly"])
@safe_plot
def contour_3d_plot(df, x_col=None, y_col=None, z_col=None, title="", engine="plotly", **kwargs):
    """
    Creates 3D contour visualization for continuous data.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if df.shape[1] < 3:
        raise ValueError("3D Contour Plot requires at least three numeric columns.")

    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    fig = go.Figure(data=[go.Surface(
        x=x.values.reshape(int(np.sqrt(len(x))), -1),
        y=y.values.reshape(int(np.sqrt(len(y))), -1),
        z=z.values.reshape(int(np.sqrt(len(z))), -1),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
    )])
    fig.update_layout(title="3D Contour Plot", scene=dict(
        xaxis_title=df.columns[0], yaxis_title=df.columns[1], zaxis_title=df.columns[2]
    ))
    return "plotly", fig

# ----------------- BATCH 12 NEW PLOTS -----------------

@register_plot('Dumbbell Plot', params=["category_col", "value1_col", "value2_col"], engines=["matplotlib"])
@safe_plot
def dumbbell_plot(df, category_col=None, value1_col=None, value2_col=None, title="", engine="matplotlib", **kwargs):
    """
    Visualizes change between two numeric variables for categories.
    Expects: Category, Value1, Value2
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if df.shape[1] < 3:
        raise ValueError("Dumbbell Plot requires at least three columns (Category, Value1, Value2).")

    cat = df.iloc[:, 0]
    val1 = df.iloc[:, 1]
    val2 = df.iloc[:, 2]

    plt.figure(figsize=(8, 5))
    for i in range(len(cat)):
        plt.plot([val1[i], val2[i]], [cat[i], cat[i]], 'o-', color='gray')
    plt.scatter(val1, cat, color='red', label=df.columns[1])
    plt.scatter(val2, cat, color='green', label=df.columns[2])
    plt.legend()
    plt.title("Dumbbell Plot")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Polar Line Chart', params=["angle_col", "value_col"], engines=["matplotlib"])
@safe_plot
def polar_line_chart(df, angle_col=None, value_col=None, title="", engine="matplotlib", **kwargs):
    """
    Creates a polar (radial) line chart.
    Expects angle and value columns.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if df.shape[1] < 2:
        raise ValueError("Polar Chart requires two numeric columns (angle, value).")

    theta = df.iloc[:, 0] * np.pi / 180
    r = df.iloc[:, 1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, r, color='teal', linewidth=2)
    ax.fill(theta, r, alpha=0.25)
    plt.title("Polar Chart")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Wind Rose Plot', params=["direction_col", "speed_col"], engines=["matplotlib"])
@safe_plot
def wind_rose_plot(df, direction_col=None, speed_col=None, title="", engine="matplotlib", **kwargs):
    """
    Wind Rose plot showing frequency distribution by direction & intensity.
    Expects: Direction (degrees) and Speed.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if df.shape[1] < 2:
        raise ValueError("Wind Rose Plot requires Direction and Speed columns.")

    dir_col, speed_col = df.columns[:2]
    bins = np.arange(0, 360, 30)
    groups = pd.cut(df[dir_col], bins=bins)
    speed_means = df.groupby(groups)[speed_col].mean()

    theta = np.deg2rad(bins[:-1])
    width = np.deg2rad(30)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, speed_means, width=width, color='coral', alpha=0.7)
    plt.title("Wind Rose Plot")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Word Frequency Tree', params=["text_col"], engines=["matplotlib"])
@safe_plot
def word_freq_tree(df, text_col=None, title="", engine="matplotlib", **kwargs):
    """
    Displays a word cloud visualization from text column.
    """
    if df.select_dtypes(exclude='number').empty:
        raise ValueError("Word Frequency Tree requires a text column.")
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    text_col = df.select_dtypes(exclude='number').columns[0]
    text_data = " ".join(df[text_col].astype(str).tolist())

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Frequency Tree (Word Cloud)")
    fig = plt.gcf()
    return "matplotlib", _fig_to_png_bytes(fig)

# ==============================
# 🍩 Clean Pie & Donut Charts (No Labels / Percentages)
# ==============================

@register_plot('Clean Pie Chart', params=[], engines=["matplotlib"])
@safe_plot
def clean_pie_chart(df, title="", engine="matplotlib", **kwargs):
    """
    Clean pie chart with automatic grouping of categories.
    No labels or numbers displayed.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    # pick a categorical column
    cat_cols = df.select_dtypes(exclude='number').columns
    if len(cat_cols) == 0:
        raise ValueError("No categorical column found – please use a column with repeated text categories.")
    
    col = cat_cols[0]
    counts = df[col].astype(str).value_counts()
    
    # if there are too many categories, show top ones
    if len(counts) > 10:
        counts = counts.head(10)

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 6)))
    colors = sns.color_palette('Set2', n_colors=len(counts))
    ax.pie(counts.values, labels=None, autopct=None, startangle=90, colors=colors)
    ax.axis('equal')
    ax.set_title(title or "Clean Pie Chart", fontsize=14)
    ax.legend(counts.index, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15), frameon=False)
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot('Clean Donut Chart', params=[], engines=["matplotlib"])
@safe_plot
def clean_donut_chart(df, title="", engine="matplotlib", **kwargs):
    """
    Clean donut chart (proportional to numeric values, no labels or text).
    Uses first column as category and second column as numeric values.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    plt.close('all')

    # ensure there are at least two columns
    if df.shape[1] < 2:
        raise ValueError("Clean Donut Chart requires two columns (Category, Value).")

    # Get columns
    category_col = df.columns[0]
    value_col = df.columns[1]

    # Convert value column to numeric (handle commas)
    df = df.copy()
    
    # Remove commas if present
    if df[value_col].dtype == 'object':
        df[value_col] = df[value_col].astype(str).str.replace(',', '')
    
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])
    
    # Remove rows with zero or negative values
    df = df[df[value_col] > 0]
    
    if df.empty:
        raise ValueError("No positive numeric data available.")

    # Plot
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 6)))
    colors = sns.color_palette('tab10', n_colors=len(df))
    ax.pie(df[value_col].values, labels=None, autopct=None, startangle=90, colors=colors,
           wedgeprops=dict(width=0.45, edgecolor='white'))
    ax.axis('equal')
    ax.set_title(title or "Clean Donut Chart", fontsize=14)
    ax.legend(df[category_col].astype(str).values, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15), frameon=False)
    
    return "matplotlib", _fig_to_png_bytes(fig)

# ----------------- BATCH 13 NEW PLOTS -----------------

@register_plot("Polar Bar Chart", params=["theta_col", "r_col"], engines=["matplotlib"])
@safe_plot
def polar_bar_chart(df, theta_col, r_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    bars = ax.bar(np.deg2rad(df[theta_col]), df[r_col], width=0.4)
    for bar in bars:
        bar.set_alpha(0.6)
    ax.set_title(title)
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Contour Plot", params=["x_col", "y_col", "z_col"], engines=["matplotlib"])
@safe_plot
def contour_plot(df, x_col, y_col, z_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    X = df[x_col].values
    Y = df[y_col].values
    Z = df[z_col].values
    fig, ax = plt.subplots()
    contour = ax.tricontourf(X, Y, Z, cmap='viridis')
    plt.colorbar(contour)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Spiral Plot", params=["value_col"], engines=["matplotlib"])
@safe_plot
def spiral_plot(df, value_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    theta = np.linspace(0, 4*np.pi, len(df))
    r = df[value_col].values
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    ax.plot(theta, r, color='purple')
    ax.set_title(title)
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Waffle Chart", params=["category_col", "value_col"], engines=["matplotlib"])
@safe_plot
def waffle_chart(df, category_col, value_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    total = df[value_col].sum()
    df['perc'] = (df[value_col]/total*100).round().astype(int)
    n_cols = 10
    waffle = np.zeros((10, n_cols))
    category_index = 0
    for i in range(10):
        for j in range(n_cols):
            waffle[i,j] = category_index
            df.loc[df.index[category_index],'perc'] -= 1
            if df.loc[df.index[category_index],'perc'] == 0:
                category_index = min(category_index+1, len(df)-1)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.matshow(waffle, cmap=cm.tab20)
    ax.set_title(title)
    ax.axis("off")
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Circular Bar Plot", params=["category_col", "value_col"], engines=["matplotlib"])
@safe_plot
def circular_bar_plot(df, category_col, value_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    N = len(df)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = df[value_col].values
    width = 2 * np.pi / N
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    bars = ax.bar(theta, radii, width=width, bottom=0.0)
    for bar in bars:
        bar.set_alpha(0.6)
    ax.set_xticks(theta)
    ax.set_xticklabels(df[category_col])
    ax.set_title(title)
    return "matplotlib", _fig_to_png_bytes(fig)


# ----------------- BATCH 14 NEW PLOTS -----------------

@register_plot("3D Scatter (Alt)", params=["x_col", "y_col", "z_col", "color_col"], engines=["matplotlib"])
@safe_plot
def scatter3d_plot(df, x_col, y_col, z_col, color_col=None, title="", engine="matplotlib", **kwargs):
    """
    3D scatter plot with optional color mapping.
    """
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    _require_columns(df, [x_col, y_col, z_col])
    _require_numeric(df, [x_col, y_col, z_col])
    
    plt.close('all')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = df[color_col] if (color_col and color_col in df.columns) else 'blue'
    ax.scatter(df[x_col], df[y_col], df[z_col], c=colors, cmap='viridis', s=50)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title or "3D Scatter Plot")
    
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("3D Surface (Alt)", params=["x_col", "y_col", "z_col"], engines=["matplotlib"])
@safe_plot
def surface3d_plot(df, x_col, y_col, z_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(df[x_col], df[y_col], df[z_col], cmap='viridis', edgecolor='none')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("3D Wireframe", params=["x_col", "y_col", "z_col"], engines=["matplotlib"])
@safe_plot
def wireframe3d_plot(df, x_col, y_col, z_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(df[x_col], df[y_col], df[z_col], cmap='plasma', linewidth=0.2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("2D Density Contour", params=["x_col", "y_col"], engines=["matplotlib"])
@safe_plot
def density_contour_plot(df, x_col, y_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig, ax = plt.subplots()
    sns.kdeplot(x=df[x_col], y=df[y_col], fill=True, cmap='viridis', ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Cluster Heatmap", params=[], engines=["matplotlib"])
@safe_plot
def cluster_heatmap(df, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig = sns.clustermap(df.corr(), cmap='coolwarm', annot=True)
    plt.title(title)
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Time Heatmap", params=["date_col", "value_col"], engines=["matplotlib"])
@safe_plot
def time_heatmap(df, date_col, value_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    df[date_col] = pd.to_datetime(df[date_col])
    df['Day'] = df[date_col].dt.day
    df['Month'] = df[date_col].dt.month
    pivot = df.pivot_table(values=value_col, index='Month', columns='Day', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, cmap='YlGnBu', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Month")
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Calendar Heatmap", params=["date_col", "value_col"], engines=["matplotlib"])
@safe_plot
def calendar_heatmap(df, date_col, value_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    df[date_col] = pd.to_datetime(df[date_col])
    df['Day'] = df[date_col].dt.day
    df['Week'] = df[date_col].dt.isocalendar().week
    df['Weekday'] = df[date_col].dt.weekday
    pivot = df.pivot_table(values=value_col, index='Week', columns='Weekday', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(pivot, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Weekday (0=Mon)")
    ax.set_ylabel("Week Number")
    return "matplotlib", _fig_to_png_bytes(fig)


# ----------------- BATCH 15 NEW PLOTS -----------------

@register_plot("Animated Line Plot", params=["x_col", "y_col"], engines=["matplotlib"])
@safe_plot
def animated_line_plot(df, x_col, y_col, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(df[x_col].min(), df[x_col].max())
    ax.set_ylim(df[y_col].min(), df[y_col].max())
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = df[x_col].iloc[:i]
        y = df[y_col].iloc[:i]
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(df), interval=100, blit=True)
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Animated Scatter", params=["x_col", "y_col", "size_col", "frame_col"], engines=["plotly"])
@safe_plot
def animated_scatter_plot(df, x_col, y_col, size_col, frame_col, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=frame_col,
                     animation_frame=frame_col, range_x=[df[x_col].min(), df[x_col].max()],
                     range_y=[df[y_col].min(), df[y_col].max()],
                     title=title, template="plotly_white")
    return "plotly", fig

@register_plot("Animated Bubble", params=["x_col", "y_col", "size_col", "color_col", "frame_col"], engines=["plotly"])
@safe_plot
def animated_bubble_chart(df, x_col, y_col, size_col, color_col, frame_col, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                     animation_frame=frame_col, hover_name=color_col, size_max=50,
                     template="plotly_white", title=title)
    return "plotly", fig

@register_plot("Interactive Funnel", params=["stage_col", "value_col"], engines=["plotly"])
@safe_plot
def interactive_funnel(df, stage_col, value_col, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig = px.funnel(df, x=value_col, y=stage_col, title=title, template="plotly_white")
    return "plotly", fig

@register_plot("Interactive Sunburst", params=["path_cols", "value_col"], engines=["plotly"])
@safe_plot
def interactive_sunburst(df, path_cols, value_col, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig = px.sunburst(df, path=path_cols, values=value_col, title=title, template="plotly_white")
    return "plotly", fig

@register_plot("Animated Geo Map", params=["lat_col", "lon_col", "size_col", "color_col", "frame_col"], engines=["plotly"])
@safe_plot
def animated_map(df, lat_col, lon_col, size_col, color_col, frame_col, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    fig = px.scatter_geo(df, lat=lat_col, lon=lon_col, size=size_col, color=color_col,
                         animation_frame=frame_col, projection="natural earth",
                         title=title, template="plotly_white")
    return "plotly", fig


# ----------------- BATCH 16 NEW PLOTS -----------------

@register_plot("Pareto Chart", params=["x", "y"], engines=["matplotlib"])
@safe_plot
def pareto_chart(df, x, y, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna().sort_values(by=y, ascending=False)
    data['cumperc'] = data[y].cumsum() / data[y].sum() * 100

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))
    ax1.bar(data[x].astype(str), data[y], color='C0')
    ax1.set_xlabel(x)
    ax1.set_ylabel(y)

    ax2 = ax1.twinx()
    ax2.plot(data[x].astype(str), data['cumperc'], color='C1', marker='o')
    ax2.set_ylabel('Cumulative %')
    ax2.axhline(80, color='red', linestyle='--', lw=1)
    ax1.set_title(title or 'Pareto Chart')
    return 'matplotlib', _fig_to_png_bytes(fig)


@register_plot("Dumbbell Chart", params=["x", "start", "end"], engines=["matplotlib", "plotly"])
@safe_plot
def dumbbell_chart(df, x, start, end, title="", engine="matplotlib", **kwargs):
    df = _auto_convert_numeric(df)
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    _require_columns(df, [x, start, end])
    _require_numeric(df, [start, end])
    data = df[[x, start, end]].dropna()

    if engine == 'plotly':
        import plotly.graph_objects as go
        theme_cfg = kwargs.get("theme_config", {})
        _apply_theme(engine, theme_cfg)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[start], y=data[x], mode='markers', name=start))
        fig.add_trace(go.Scatter(x=data[end], y=data[x], mode='markers', name=end))
        for i in range(len(data)):
            fig.add_shape(type='line', x0=data[start].iloc[i], x1=data[end].iloc[i],
                          y0=data[x].iloc[i], y1=data[x].iloc[i], line=dict(color='gray'))
        fig.update_layout(title=title or 'Dumbbell Chart')
        return 'plotly', fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))
    ax.hlines(y=data[x], xmin=data[start], xmax=data[end], color='gray')
    ax.plot(data[start], data[x], 'o', label=start)
    ax.plot(data[end], data[x], 'o', label=end)
    ax.legend()
    ax.set_title(title or 'Dumbbell Chart')
    return 'matplotlib', _fig_to_png_bytes(fig)


@register_plot("Bullet Chart", params=["category", "value", "target"], engines=["plotly"])
@safe_plot
def bullet_chart(df, category, value, target, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [category, value, target])
    _require_numeric(df, [value, target])
    data = df[[category, value, target]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    import plotly.graph_objects as go
    fig = go.Figure()
    for _, row in data.iterrows():
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=row[value],
            gauge={"shape": "bullet", "axis": {"range": [0, max(data[target])*1.2]},
                   "threshold": {"value": row[target], "line": {"color": "red", "width": 2}}},
            title={"text": str(row[category])}
        ))
    fig.update_layout(title=title or 'Bullet Chart', height=300*len(data))
    return 'plotly', fig


@register_plot("Polar Area Chart", params=["category", "value"], engines=["plotly"])
@safe_plot
def polar_area_chart(df, category, value, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [category, value])
    _require_numeric(df, [value])
    data = df[[category, value]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    import plotly.express as px
    fig = px.bar_polar(data, r=value, theta=category, title=title or 'Polar Area Chart')
    return 'plotly', fig


@register_plot("Marimekko Chart", params=["x", "y", "width"], engines=["plotly"])
@safe_plot
def marimekko_chart(df, x, y, width, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y, width])
    _require_numeric(df, [y, width])
    data = df[[x, y, width]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    import plotly.express as px
    fig = px.bar(data, x=x, y=y, width=data[width]*100, title=title or 'Marimekko Chart')
    return 'plotly', fig

# ----------------- BATCH 17 NEW PLOTS -----------------

@register_plot("Polar Scatter Plot", params=["r", "theta"], engines=["plotly", "matplotlib"])
@safe_plot
def polar_scatter(df, r, theta, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [r, theta])
    _require_numeric(df, [r])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.express as px
        fig = px.scatter_polar(df, r=r, theta=theta, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.scatter(np.deg2rad(df[theta]), df[r])
    ax.set_title(title or "Polar Scatter Plot")
    return "matplotlib", _fig_to_png_bytes(fig)

@register_plot("Lollipop Chart", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def lollipop_chart(df, x, y, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    data = df[[x, y]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[x], y=data[y], mode="lines+markers"))
        fig.update_layout(title=title or "Lollipop Chart")
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.vlines(x=data[x], ymin=0, ymax=data[y], color="skyblue", linewidth=2)
    ax.scatter(data[x], data[y], color="blue", s=80)
    ax.set_title(title or f"Lollipop Chart of {y} by {x}")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Error Bar Chart", params=["x", "y", "error"], engines=["matplotlib", "plotly"])
@safe_plot
def error_bar_chart(df, x, y, error, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y, error])
    _require_numeric(df, [y, error])
    data = df[[x, y, error]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Scatter(
                x=data[x], y=data[y],
                error_y=dict(type='data', array=data[error]),
                mode='markers+lines'
            )
        ])
        fig.update_layout(title=title or "Error Bar Chart")
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(data[x], data[y], yerr=data[error], fmt='o-', capsize=5)
    ax.set_title(title or f"Error Bar Chart of {y} by {x}")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Dot Plot", params=["x", "y"], engines=["matplotlib", "plotly"])
@safe_plot
def dot_plot(df, x, y, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y])
    _require_numeric(df, [y])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.express as px
        fig = px.strip(df, x=x, y=y, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[y], df[x], 'o')
    ax.set_title(title or "Dot Plot")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Pyramid Chart", params=["x", "y", "group"], engines=["plotly", "matplotlib"])
@safe_plot
def pyramid_chart(df, x, y, group, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y, group])
    _require_numeric(df, [y])
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    data = df.dropna(subset=[x, y, group])
    g1, g2 = data[group].unique()[:2]

    if engine == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(y=data[data[group] == g1][x], x=data[data[group] == g1][y], name=str(g1), orientation='h'))
        fig.add_trace(go.Bar(y=data[data[group] == g2][x], x=-data[data[group] == g2][y], name=str(g2), orientation='h'))
        fig.update_layout(title=title or "Pyramid Chart", barmode='overlay')
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data[data[group] == g1][x], data[data[group] == g1][y], color='blue', label=g1)
    ax.barh(data[data[group] == g2][x], -data[data[group] == g2][y], color='orange', label=g2)
    ax.legend()
    ax.set_title(title or "Pyramid Chart")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Spider Chart", params=["category", "values"], engines=["plotly", "matplotlib"])
@safe_plot
def spider_chart(df, category, values, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [category] + values)
    _require_numeric(df, values)
    data = df[[category] + values].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        for _, row in data.iterrows():
            fig.add_trace(go.Scatterpolar(r=row[values], theta=values, fill="toself", name=row[category]))
        fig.update_layout(title=title or "Spider Chart", polar=dict(radialaxis=dict(visible=True)))
        return "plotly", fig

    import matplotlib.pyplot as plt
    from math import pi
    categories = values
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
    for _, row in data.iterrows():
        vals = row[values].tolist() + [row[values[0]]]
        ax.plot(angles, vals, linewidth=1, linestyle='solid', label=row[category])
        ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_title(title or "Spider Chart")
    return "matplotlib", _fig_to_png_bytes(fig)


@register_plot("Slope Chart", params=["x", "y", "group"], engines=["matplotlib", "plotly"])
@safe_plot
def slope_chart(df, x, y, group, title="", engine="plotly", **kwargs):
    df = _auto_convert_numeric(df)
    _require_columns(df, [x, y, group])
    _require_numeric(df, [y])
    data = df[[x, y, group]].dropna()
    theme_cfg = kwargs.get("theme_config", {})
    _apply_theme(engine, theme_cfg)
    if engine == "plotly":
        import plotly.express as px
        fig = px.line(data, x=x, y=y, color=group, markers=True, title=title)
        return "plotly", fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group_df in data.groupby(group):
        ax.plot(group_df[x], group_df[y], marker="o", label=label)
    ax.legend()
    ax.set_title(title or f"Slope Chart of {y} over {x}")
    return "matplotlib", _fig_to_png_bytes(fig)

