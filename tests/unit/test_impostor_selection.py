"""Which LFW names the impostor-tail benchmark reads as East Asian, and which it sets aside."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_impostor_tails as tails  # noqa: E402


@pytest.mark.parametrize(
    "name",
    [
        # Two-syllable given names written as one word, which the old rule missed.
        "Li_Changchun", "Zhang_Yimou", "Zeng_Qinghong", "Jia_Qinglin",
        # Korean names, hyphenated or not, including family names the old list lacked.
        "Kim_Dae-jung", "Jeong_Se-hyun", "Jung_Bong", "Chan_Ho_Park", "Sun_Myung_Moon",
        # An English name before a Chinese one.
        "Alan_Tang_Kwong-wing", "Vicki_Zhao_Wei",
        # Japanese given names in Hepburn before a Japanese family name.
        "Junichiro_Koizumi", "Hidetoshi_Nakata",
        # Western given names the manual list vouches for.
        "Michelle_Yeoh", "Jackie_Chan",
    ],
)
def test_east_asian_names_are_selected(name: str) -> None:
    assert tails.population(name) == "east_asian"


@pytest.mark.parametrize(
    "name",
    ["Kim_Clijsters", "Lee_Ann_Womack", "Jamie_Lee_Curtis", "Spike_Lee", "Lee_Baca",
     "Jo_Dee_Messina", "Mary_Lou_Retton", "Mika_Hakkinen", "Jose_Mourinho"],
)
def test_other_names_are_not(name: str) -> None:
    assert tails.population(name) == "other"


@pytest.mark.parametrize("name", ["Kate_Lee", "Ronald_Ito", "Young_Kim", "Stanley_Ho"])
def test_names_the_rules_cannot_place_join_no_group(name: str) -> None:
    """A Western given name before an East Asian family name stays out of the controls too."""
    assert tails.population(name) == "ambiguous"
