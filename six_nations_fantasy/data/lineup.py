"""Lineup status from historical match sheets."""

import pandas as pd


def substitute_status(performances: pd.DataFrame) -> pd.Series | None:
    """Read the recorded bench flag or derive it from the lineup number."""
    if "is_substitute" in performances:
        return performances["is_substitute"].astype("boolean")
    if "number" in performances:
        numbers = pd.to_numeric(performances["number"], errors="coerce")
        return numbers.gt(15).astype("boolean").where(numbers.ge(1))
    return None
