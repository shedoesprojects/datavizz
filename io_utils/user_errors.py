def format_user_error(error: Exception) -> dict:
    """
    Convert technical Python errors into user-friendly messages.
    Returns a dict with title + message.
    """

    msg = str(error).lower()

    # Missing columns
    if "missing required column" in msg:
        return {
            "title": "Required data column not found",
            "message": (
                "This chart needs specific columns that are not present in your data.\n\n"
                "Please check your column selections and try again."
            )
        }

    # Numeric issues
    if "must be numeric" in msg or "numeric" in msg:
        return {
            "title": "Numeric data required",
            "message": (
                "This chart works only with numeric values.\n\n"
                "Please choose columns that contain numbers like sales, score, or count."
            )
        }

    # Empty data after filtering
    if "no data" in msg or "empty" in msg:
        return {
            "title": "No usable data to display",
            "message": (
                "After applying the selected options, there is no data left to plot.\n\n"
                "Try selecting different columns or reviewing missing values."
            )
        }

    # Date parsing issues
    if "date" in msg or "datetime" in msg:
        return {
            "title": "Invalid date column",
            "message": (
                "This chart expects a date or time-based column.\n\n"
                "Please select a column that contains valid dates."
            )
        }

    # Fallback (VERY IMPORTANT)
    return {
        "title": "Chart could not be generated",
        "message": (
            "We couldn’t create this chart with the selected settings.\n\n"
            "Try adjusting the chart type or selected columns."
        )
    }
