"""Overlap guard must FAIL LOUDLY on planted overlaps and pass on clean data.
Run: python -m tests.test_overlap   (needs results/tier2/mmstar_fingerprints.json from a Tier 2 run)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.overlap import assert_no_overlap

fp = json.load(open("results/tier2/mmstar_fingerprints.json"))
long_q = next(q for q in fp["q"] if len(q) >= 30)
far = int(fp["dhash"][0]) ^ ((1 << 64) - 1)  # all 64 bits flipped: far from image 0 (checked against all below)


def raises(ids, hashes, qs):
    try:
        assert_no_overlap(ids, hashes, qs, fp, 4, "planted")
    except AssertionError as e:
        return str(e)[:110]
    return None


# 1. image near-duplicate: exact copy and a copy with 3 flipped bits
h0 = int(fp["dhash"][5])
assert raises(["t-1"], [h0], ["zzz unrelated question text here that is long enough"]), "exact image copy not caught"
assert raises(["t-1"], [h0 ^ 0b111], ["zzz unrelated question text here that is long enough"]), "near-duplicate image not caught"
# 2. identical long question text
assert raises(["t-1"], [far], [long_q]), "identical question not caught"
# 3. id collision
assert raises(["mmstar-0"], [far], ["zzz unrelated question text here that is long enough"]), "id collision not caught"
# 4. clean item passes (pick a hash far from every eval image)
cand = next(h for h in (0x0123456789ABCDEF, 0xF0F0F0F00F0F0F0F, 0xAAAAAAAA55555555)
            if all((h ^ int(e)).bit_count() > 4 for e in fp["dhash"]))
assert raises(["t-clean"], [cand], ["zzz unrelated question text here that is long enough"]) is None, "clean item wrongly flagged"
print("overlap guard tests passed: exact copy, near-duplicate, same question, id collision all raise; clean item passes")
print("example message:", raises(["t-1"], [h0 ^ 0b111], ["zzz unrelated question text here that is long enough"]))
