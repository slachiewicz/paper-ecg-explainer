"""End-to-end checks on fictional scans drawn by demo/make_demo.py (no real patient data)."""
import os
import re

import cv2
import numpy as np
import pytest

import ecg_explain
import make_demo
import report
import viewer
from analysis import analyze, atrial_spectrum, find_anomalies
from digitize import DigitizeError, digitize

LIMB = "I,II,III,aVR,aVL,aVF".split(",")
CHEST = "V1,V2,V3,V4,V5,V6".split(",")


def _scans(folder, sinus_rate=None):
    os.makedirs(folder, exist_ok=True)
    return (make_demo.make_scan(LIMB, "limb.png", 7, sinus_rate, folder),
            make_demo.make_scan(CHEST, "chest.png", 11, sinus_rate, folder))


def _load(limb, chest):
    return [digitize(limb, LIMB), digitize(chest, CHEST)]


@pytest.fixture(scope="module")
def af(tmp_path_factory):
    return _scans(str(tmp_path_factory.mktemp("af")))


@pytest.fixture(scope="module")
def sinus(tmp_path_factory):
    return _scans(str(tmp_path_factory.mktemp("sinus")), sinus_rate=72)


def test_fibrillation_is_recognized(af):
    r = analyze(_load(*af))
    assert r.conclusion == "af"
    assert r.irregularity == "high"
    assert r.p_waves == "absent"
    assert r.atrial_waves == "fibrillation"
    anomalies, waves, _ = find_anomalies(r)
    assert any(a["type"] == "fWaves" for a in anomalies)
    assert waves


def test_normal_rhythm_has_no_fibrillation_signs(sinus):
    r = analyze(_load(*sinus))
    assert r.conclusion == "sinus"
    assert 68 < r.heart_rate < 76
    assert r.p_waves == "repeatable"
    assert r.atrial_waves == "none"
    anomalies, waves, _ = find_anomalies(r)
    assert not [a for a in anomalies if a["type"] in ("fWaves", "irregular", "noP")]
    assert not waves


@pytest.mark.parametrize("rate", [40, 50])
def test_slow_rhythm_is_not_called_fibrillation(tmp_path, rate):
    r = analyze(_load(*_scans(str(tmp_path), sinus_rate=rate)))
    assert r.conclusion in ("sinus", "unclear")
    anomalies, waves, _ = find_anomalies(r)
    assert not [a for a in anomalies if a["type"] == "fWaves"]


def test_scans_with_different_resolutions(af, tmp_path):
    limb, chest = af
    small = str(tmp_path / "chest_small.png")
    cv2.imwrite(small, cv2.resize(cv2.imread(chest), None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA))
    printouts = _load(limb, small)
    assert printouts[0].fs != pytest.approx(printouts[1].fs, rel=0.1)
    r = analyze(printouts)
    alone = [atrial_spectrum([w for w in r.windows if w.leads[0].name in names])[2] for names in (LIMB, CHEST)]
    assert r.dominant_freq == pytest.approx(np.mean(alone), rel=0.05)
    assert r.atrial_waves == "fibrillation"


def test_single_picture(af):
    r = analyze([digitize(af[0], LIMB)])
    assert len(r.windows) == 2
    assert r.conclusion == "af"


def test_analyze_does_not_change_the_printouts(af):
    printouts = _load(*af)
    first = analyze(printouts)
    second = analyze(printouts)
    assert np.array_equal(first.rr, second.rr)
    assert len(printouts[0].leads[0].mv) > len(first.windows[0].leads[0].mv)


def test_greyscale_copy_gives_a_clear_error(af, tmp_path):
    grey = str(tmp_path / "grey.png")
    cv2.imwrite(grey, cv2.cvtColor(cv2.cvtColor(cv2.imread(af[0]), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
    with pytest.raises(DigitizeError) as e:
        digitize(grey, LIMB)
    assert e.value.code == "no_grid"


def test_every_error_has_a_message_in_every_language():
    for lang in ("en", "pl"):
        for code in ("no_grid", "grid_unclear", "resolution", "rows", "no_beats"):
            assert ecg_explain.CONSOLE[lang][code].format(f="x", found=1, expected=3)


def test_lead_names_are_case_insensitive():
    assert [ecg_explain.CANONICAL_LEADS[x.lower()] for x in ("ii", "AVF", "v1")] == ["II", "aVF", "V1"]


@pytest.mark.parametrize("lang", ["en", "pl"])
def test_report_and_viewer_without_nan(tmp_path, lang):
    # at 50/min there are too few beats per segment for the P-wave similarity, which is then NaN
    printouts = _load(*_scans(str(tmp_path / "scans"), sinus_rate=50))
    r = analyze(printouts)
    assert np.isnan(r.p_correlation)
    report.generate(printouts, r, str(tmp_path / "report.html"), lang)
    viewer.generate(printouts, r, str(tmp_path / "viewer.html"), lang)
    assert "/*__DATA__*/null" not in (tmp_path / "viewer.html").read_text(encoding="utf-8")
    text = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "", (tmp_path / "report.html").read_text(encoding="utf-8"))
    assert not re.search(r"\bnan\b", text, re.I)
