"""Match spreadsheet spellings to RugbyPass player names."""

import re
import unicodedata

NAME_ALIASES = {
    ("Ireland", "tomclarkson"): "thomasclarkson",
    ("Ireland", "tomfarrell"): "thomasfarrell",
    ("Italy", "nachobrex"): "juanignaciobrex",
    ("France", "noahnene"): "noahtisienene",
    ("Scotland", "freddydouglas"): "freddiedouglas",
    ("Wales", "gabehamerwebb"): "gabrielhamerwebb",
    ("Wales", "danedwards"): "danieledwards",
    ("Italy", "davidodiase"): "davideodiase",
    ("England", "finsmith"): "finlaysmith",
    ("England", "sebatkinson"): "sebastienatkinson",
}


def name_key(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def add_name_keys(frame):
    return frame.assign(name_key=[
        NAME_ALIASES.get((team, key), key)
        for team, key in zip(frame["team"], frame["name"].map(name_key))
    ])
