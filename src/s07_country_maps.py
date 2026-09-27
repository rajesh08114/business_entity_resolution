"""Add the hard-coded country maps (normalize.COUNTRY_*_MAPS) to an existing work folder without relearning the others.
Used to patch a finished run; a fresh run gets them from s01_lexicon.py."""
import json

from common import WORK_DIR
from normalize import COUNTRY_ADDR_MAPS, COUNTRY_NAME_MAPS

if __name__ == "__main__":
    for fname, extra_maps in (("address_map.json", COUNTRY_ADDR_MAPS), ("name_map.json", COUNTRY_NAME_MAPS)):
        path = WORK_DIR / fname
        maps = json.load(open(path, encoding="utf-8"))
        for country, extra in extra_maps.items():
            maps[country] = {**maps.get(country, {}), **extra}
        json.dump(maps, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(fname, {c: len(v) for c, v in maps.items()})
