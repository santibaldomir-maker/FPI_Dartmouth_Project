"""
test_scoring.py — Physics-consistency and outcome-independence tests for score_fpi().

Key invariants:
  1. Outcome-independence: identical physics → identical score regardless of outcome words.
  2. Correct FPI level by mode and height/voltage.
  3. Entrapment modifier forces s ≤ 2 (physics signal, not outcome).
  4. Compound-hazard modifier forces s ≤ 2.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from dtf_platform import score_fpi


# ── Helpers ──────────────────────────────────────────────────────────────────

def score(fm, h=np.nan, v=np.nan, narr=''):
    s, _ = score_fpi(fm, h, v, narr)
    return s


# ── Outcome-independence ──────────────────────────────────────────────────────

OUTCOME_WORDS = ['fatal', 'fatality', 'killed', 'died', 'death', 'serious injury',
                 'no injury', 'near miss', 'first aid', 'hospitalized']

@pytest.mark.parametrize('fm,h,v,narr', [
    ('FG-HOLE',  np.nan, np.nan, 'fell through skylight'),
    ('SLL',      np.nan, np.nan, 'crane load swung'),
    ('EEE',      np.nan, 480,    'contacted power line'),
    ('EOE',      np.nan, np.nan, 'struck by forklift'),
    ('CCZ',      np.nan, np.nan, 'full body caught between'),
    ('ERG',      np.nan, np.nan, 'repetitive strain lifting'),
    ('FG-SCAF',  20,     np.nan, 'fell from scaffold'),
    ('FG-VEH',   np.nan, np.nan, 'fell from truck step'),
])
def test_outcome_independence(fm, h, v, narr):
    """Adding outcome words must not change the FPI score."""
    base = score(fm, h, v, narr)
    for word in OUTCOME_WORDS:
        modified = score(fm, h, v, narr + ' ' + word)
        assert modified == base, (
            f"{fm}: score changed from {base} → {modified} when adding '{word}'"
        )


# ── Correct FPI levels by mode ────────────────────────────────────────────────

class TestFallGeometry:
    def test_fg_hole_always_1(self):
        assert score('FG-HOLE') == 1

    def test_fg_lift_always_1(self):
        assert score('FG-LIFT', h=30) == 1

    def test_fg_lift_low_always_1(self):
        # height floor removed — any aerial lift fall is fatal-physics
        assert score('FG-LIFT', h=8) == 1

    def test_fg_roof_high_is_1(self):
        assert score('FG-ROOF', h=20) == 1

    def test_fg_roof_low_always_1(self):
        # height floor removed — roof edge fall always fatal-physics
        assert score('FG-ROOF', h=3) == 1

    def test_fg_scaf_high_is_1(self):
        assert score('FG-SCAF', h=10) == 1

    def test_fg_scaf_low_always_1(self):
        # height floor removed — scaffold fall always fatal-physics
        assert score('FG-SCAF', h=4) == 1

    def test_fg_lad_always_1(self):
        assert score('FG-LAD', h=10) == 1

    def test_fg_str_always_1(self):
        assert score('FG-STR', h=20) == 1

    def test_fg_veh_default_is_2(self):
        assert score('FG-VEH') == 2

    def test_fg_veh_height_escalates_to_1(self):
        assert score('FG-VEH', h=6) == 1

    def test_fg_gen_default_is_2(self):
        assert score('FG-GEN') == 2

    def test_fg_gen_height_escalates_to_1(self):
        assert score('FG-GEN', h=10) == 1


class TestHighEnergyModes:
    def test_sll_always_1(self):
        assert score('SLL') == 1

    def test_sme_always_1(self):
        assert score('SME') == 1

    def test_lfr_default_1(self):
        assert score('LFR') == 1

    def test_lfr_deflected_is_2(self):
        assert score('LFR', narr='deflected off wall partial exposure') == 2

    def test_eee_high_voltage_is_1(self):
        assert score('EEE', v=480) == 1

    def test_eee_power_line_narr_is_1(self):
        assert score('EEE', narr='contacted overhead power line') == 1

    def test_eee_default_is_2(self):
        assert score('EEE') == 2

    def test_eoe_default_is_1(self):
        assert score('EOE') == 1

    def test_vpt_always_1(self):
        assert score('VPT') == 1

    def test_mei_rollover_is_1(self):
        assert score('MEI', narr='forklift rollover with operator') == 1

    def test_mei_near_instability_is_2(self):
        assert score('MEI', narr='equipment became unstable near tip') == 2

    def test_cee_default_is_1(self):
        assert score('CEE') == 1

    def test_cee_controlled_entry_is_2(self):
        assert score('CEE', narr='entered confined space no hazard detected monitoring completed') == 2

    def test_hta_default_is_1(self):
        assert score('HTA') == 1

    def test_hta_sub_lethal_is_2(self):
        assert score('HTA', narr='sub-lethal exposure detected by monitoring headache only') == 2

    def test_fet_default_is_1(self):
        assert score('FET') == 1

    def test_fet_small_fire_is_2(self):
        assert score('FET', narr='small contained fire quickly extinguished') == 2


class TestMediumLowModes:
    def test_ogp_default_is_1(self):
        # construction default: overhead objects are heavy materials
        assert score('OGP') == 1

    def test_ogp_heavy_load_is_1(self):
        assert score('OGP', narr='worker directly below heavy load dropped') == 1

    def test_ogp_light_object_is_3(self):
        assert score('OGP', narr='small bolt fell from scaffold') == 3

    def test_iea_default_is_2(self):
        assert score('IEA') == 2

    def test_iea_height_is_1(self):
        assert score('IEA', h=10) == 1

    def test_ccz_default_is_1(self):
        # machinery has enormous mechanical advantage — default is fatal-physics
        assert score('CCZ') == 1

    def test_ccz_extremity_only_is_2(self):
        # isolated extremity pinch with light component de-escalates to medium
        assert score('CCZ', narr='finger pinched in light hinge') == 2

    def test_ccz_full_body_is_1(self):
        assert score('CCZ', narr='full body crushed between') == 1

    def test_tfe_default_is_2(self):
        assert score('TFE') == 2

    def test_tfe_heat_stroke_is_1(self):
        assert score('TFE', narr='heat stroke victim collapsed') == 1

    def test_tfe_mild_is_3(self):
        assert score('TFE', narr='mild heat exposure slight discomfort') == 3

    def test_erg_always_3(self):
        assert score('ERG') == 3


# ── Entrapment modifier ───────────────────────────────────────────────────────

class TestEntraptmentModifier:
    def test_trapped_escalates_to_max_2(self):
        # ERG base = 3, entrapment → ≤2
        assert score('ERG', narr='worker trapped in machine') <= 2

    def test_pinned_escalates(self):
        assert score('OGP', narr='worker pinned under fallen debris') <= 2

    def test_buried_escalates(self):
        assert score('FG-GEN', narr='worker buried in trench collapse') == 1

    def test_engulfed_escalates(self):
        assert score('CEE', narr='worker engulfed in grain') == 1

    def test_no_entrapment_no_change(self):
        assert score('ERG', narr='repetitive motion shoulder pain') == 3


# ── Score is always 1, 2, or 3 ───────────────────────────────────────────────

ALL_CODES = [
    'FET','CEE','SLL','SME','LFR','EEE','EOE','VPT','MEI','OGP',
    'IEA','CCZ','HTA','TFE','FG-HOLE','FG-LIFT','FG-ROOF','FG-SCAF',
    'FG-LAD','FG-STR','FG-VEH','FG-GEN','ERG','SBO','OTHER',
]

@pytest.mark.parametrize('fm', ALL_CODES)
def test_score_always_in_range(fm):
    s = score(fm, h=5, v=120, narr='worker injured during operation')
    assert s in (1, 2, 3), f"{fm}: score={s} outside 1-3 range"
