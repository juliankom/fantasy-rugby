"""Match spreadsheet spellings to RugbyPass player names."""

import re
import unicodedata

NAME_ALIASES = {
    ("Ireland", "tomclarkson"): "thomasclarkson",
    ("Ireland", "tomfarrell"): "thomasfarrell",
    ("Italy", "nachobrex"): "juanignaciobrex",
    ("Italy", "davidodiase"): "davideodiase",
    ("Italy", "sebastiannegri"): "sebastiannegridaollegio",
    ("France", "noahnene"): "noahtisienene",
    ("Scotland", "freddydouglas"): "freddiedouglas",
    ("England", "finsmith"): "finlaysmith",
    ("England", "sebatkinson"): "sebastienatkinson",
    ("Wales", "elliotdee"): "elliottdee",
    ("Wales", "keironassiratti"): "kieronassiratti",
    ("Wales", "gabehamerwebb"): "gabrielhamerwebb",
    ("Wales", "danedwards"): "danieledwards",
    ("Wales", "willgriffjohn"): "willjohn",
}


def name_key(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def add_name_keys(frame):
    return frame.assign(name_key=[
        NAME_ALIASES.get((team, key), key)
        for team, key in zip(frame["team"], frame["name"].map(name_key))
    ])
