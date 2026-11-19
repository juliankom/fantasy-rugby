"""Read the same JSON that RugbyPass supplies to its match-centre app."""

import json
import re
from urllib.parse import parse_qs, urlsplit


def embedded_json(html, field):
    match = re.search(r"\b" + re.escape(field) + r":\s*(?=[{\[])", html)
    if match is None:
        raise ValueError(f"RugbyPass page is missing {field!r}")
    return json.JSONDecoder().raw_decode(html, match.end())[0]


def match_id(url):
    values = parse_qs(urlsplit(url).query).get("g", [])
    if len(values) != 1 or not values[0].isdigit():
        raise ValueError(f"Expected one numeric match ID in {url}")
    return int(values[0])
