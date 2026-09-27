"""Step 1: learn token maps from the TRAIN labels (no external resources).

lexicon.json      non-Latin -> Latin name tokens (all train pairs)
address_map.json  per country: non-Latin address tokens (e.g. state names in Devanagari) and spelling variants / aliases
                  (poona->pune, srteet->street). Learned from train entities outside the evaluation hold-out
                  (same hash split as s03_train.py) so that the held-out evaluation stays unbiased.
"""
import json

import numpy as np
import pandas as pd

from common import DATA_DIR, WORK_DIR, id_num, read_split
from normalize import COUNTRY_ADDR_MAPS, COUNTRY_NAME_MAPS, learn_alias, learn_lexicon, norm_addr, norm_name

HFRAC = 0.15
MAX_ADDR_PAIRS = 600_000


def u01(nums):
    return ((nums.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32)).astype(np.float64) / 2 ** 32


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    d = read_split("train")
    gt = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.matched_entity_ids != ""]
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")[["source1_entity_id", "m"]]
    s1 = d["s1"].set_index("entity_id")
    b = pd.concat([d["s2"].set_index("entity_id"), d["s3"].set_index("entity_id")])

    nb_raw = pairs.m.map(b.business_name)
    nl = pairs[~nb_raw.fillna("").map(str.isascii)]
    n1 = [norm_name(x) for x in nl.source1_entity_id.map(s1.business_name)]
    nb = [norm_name(x) for x in nl.m.map(b.business_name)]
    print("non-Latin name pairs:", len(nl))
    lex = learn_lexicon(n1, nb, min_j=0.0)
    print("name lexicon size:", len(lex))
    with open(WORK_DIR / "lexicon.json", "w", encoding="utf-8") as f:
        json.dump(lex, f, ensure_ascii=False)

    u = u01(id_num(pairs.source1_entity_id))
    outside = pairs[~((u >= HFRAC * 0.7) & (u < HFRAC))]
    outside = outside.assign(country=outside.source1_entity_id.map(s1.country))
    maps, name_maps = {}, {}
    for country, g in outside.groupby("country"):
        g = g.sample(min(len(g), MAX_ADDR_PAIRS), random_state=0)
        n1 = [norm_name(x) for x in g.source1_entity_id.map(s1.business_name)]
        nb = [norm_name(x) for x in g.m.map(b.business_name)]
        name_maps[country] = learn_alias(n1, nb, synonym_prec=1.01)  # spelling variants only: pivate->private, lndia->india
        print(f"{country}: name spelling variants {len(name_maps[country])}: " + ", ".join(f"{k}->{v}" for k, v in name_maps[country].items()))
        a1 = [norm_addr(x) for x in g.source1_entity_id.map(s1.business_address)]
        ab = [norm_addr(x) for x in g.m.map(b.business_address)]
        keep = [i for i, (x, y) in enumerate(zip(a1, ab)) if x and y]
        a1, ab = [a1[i] for i in keep], [ab[i] for i in keep]
        addr_lex = learn_lexicon(a1, ab, min_j=0.0)
        alias = learn_alias(a1, ab)
        maps[country] = {**addr_lex, **alias}
        print(f"{country}: address lexicon {len(addr_lex)}, aliases {len(alias)}: " + ", ".join(f"{k}->{v}" for k, v in alias.items()))
    for country, extra in COUNTRY_ADDR_MAPS.items():
        maps[country] = {**maps.get(country, {}), **extra}
    for country, extra in COUNTRY_NAME_MAPS.items():
        name_maps[country] = {**name_maps.get(country, {}), **extra}
    with open(WORK_DIR / "address_map.json", "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=1)
    with open(WORK_DIR / "name_map.json", "w", encoding="utf-8") as f:
        json.dump(name_maps, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
