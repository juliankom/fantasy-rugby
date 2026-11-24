from unittest.mock import patch

import numpy as np
import pandas as pd

from six_nations_fantasy.data import RugbyDataset
from six_nations_fantasy.models import RandomForestPoints


def test_recent_scores_exclude_same_day_and_use_last_five_appearances():
    matches = pd.DataFrame({
        "match_id": range(8),
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03",
                                "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07"]),
    })
    rows = pd.DataFrame({"match_id": range(8), "player_id": 11, "points": range(8)})
    model = RandomForestPoints(scorer=lambda rows: rows["points"])
    with patch.object(model.forest, "fit", wraps=model.forest.fit) as fit:
        model.fit(RugbyDataset(matches, rows.iloc[::-1]))
        features, targets = fit.call_args.args
        np.testing.assert_allclose(features["points"], [np.nan, np.nan, 0.5, 1, 1.5, 2, 3, 4])
        assert features["appearances"].tolist() == [0, 0, 2, 3, 4, 5, 6, 7]
        assert targets.tolist() == list(range(8))

    fixtures = pd.DataFrame({"player_id": [99, 11], "number": [18, 4]}, index=[13, 6])
    with patch.object(model.forest, "predict", wraps=model.forest.predict) as predict:
        result = model.predict(fixtures)
        features = predict.call_args.args[0]
        np.testing.assert_allclose(features["points"], [np.nan, 5])
        assert features["is_substitute"].tolist() == [1, 0]
        assert features["appearances"].tolist() == [0, 8]
    assert result.index.equals(fixtures.index)
    assert np.isfinite(result).all()
