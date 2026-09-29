"""
Rhythm analysis of the digitized traces.

What is measured (and why):
  * heartbeats (QRS complexes) and the intervals between them (RR) -> is the rhythm regular,
  * repeatability of the P wave before each beat -> is the sinus node driving the rhythm,
  * frequency of the baseline undulation between beats -> atrial fibrillation vs. flutter.

Internal codes are English; user-facing (Polish) wording lives in `texts.py`.
"""
import copy
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

import texts

# leads in which atrial activity (P waves, f/F waves) is usually best visible
ATRIAL_LEADS = ["II", "III", "aVF", "V1"]


class NotEnoughBeats(ValueError):
    """Fewer than two heartbeats were found in every segment, so no interval can be measured."""


@dataclass
class Window:
    """A few seconds of recording in which several leads were captured simultaneously."""
    name: str
    leads: list                         # Lead objects from the same time segment
    beats: np.ndarray = None            # R-peak times [s]
    main: object = None                 # lead with the most prominent QRS


@dataclass
class Result:
    windows: list
    rr: np.ndarray                      # all RR intervals [s]
    rr_window: list                     # window index of each RR interval
    heart_rate: float
    cv: float
    nrmssd: float
    pct_large_changes: float
    irregularity: str                   # "high" / "moderate" / "low" / "unknown"
    p_correlation: float                # how similar the segments before each QRS are
    p_waves: str                        # "repeatable" / "absent" / "uncertain"
    spectrum_f: np.ndarray = None
    spectrum_p: np.ndarray = None
    dominant_freq: float = float("nan")
    organization_index: float = float("nan")
    atrial_waves: str = ""              # "fibrillation" / "flutter" / "ambiguous" / "none" / "no_data"
    p_examples: list = field(default_factory=list)   # (lead name, segments) for plotting
    conclusion: str = ""                # "af" / "flutter" / "sinus" / "unclear"


def _trim_edges(lead, sec=0.08):
    """The printer draws a vertical stroke at column edges - that is not a heart signal."""
    n = int(sec * lead.fs)
    for attr in ("t", "mv", "x_px", "y_px"):
        setattr(lead, attr, getattr(lead, attr)[n:-n])


def _slope_energy(x, fs):
    d = np.gradient(x) * fs
    e = np.convolve(d * d, np.ones(int(0.08 * fs)) / int(0.08 * fs), mode="same")
    return e / (np.percentile(e, 99) + 1e-9)


def detect_beats(window):
    """Fuse all leads of a window: a QRS is where the trace is steep in many leads at once."""
    fs = window.leads[0].fs
    t0 = max(l.t[0] for l in window.leads)
    t1 = min(l.t[-1] for l in window.leads)
    grid_t = np.arange(t0, t1, 1 / fs)
    total = np.zeros_like(grid_t)
    for l in window.leads:
        total += np.interp(grid_t, l.t, _slope_energy(l.mv, fs))
    peaks, _ = sps.find_peaks(total, distance=int(0.27 * fs), height=0.35 * np.percentile(total, 99))

    # most prominent lead = largest QRS amplitude
    window.main = max(window.leads, key=lambda l: np.percentile(np.abs(l.mv), 99.5))
    m = window.main
    r = []
    for p in peaks:
        tc = grid_t[p]
        sel = (m.t > tc - 0.07) & (m.t < tc + 0.07)
        if not sel.any():
            continue
        i = np.flatnonzero(sel)[np.argmax(np.abs(m.mv[sel]))]
        if m.t[0] + 0.04 < m.t[i] < m.t[-1] - 0.04:
            r.append(m.t[i])
    window.beats = np.array(r)


def _pre_qrs_segments(lead, beats, start=-0.30, end=-0.06):
    n0, n1 = int(start * lead.fs), int(end * lead.fs)
    segs = []
    for tr in beats:
        i = int(np.argmin(np.abs(lead.t - tr)))
        if i + n0 >= 0 and i + n1 <= len(lead.mv):
            s = lead.mv[i + n0:i + n1]
            segs.append(s - s.mean())
    return np.array(segs)


def assess_p_waves(windows):
    """If every QRS is preceded by the same P wave, the segments before each QRS look alike."""
    correlations, examples = [], []
    for w in windows:
        for l in w.leads:
            if l.name not in ATRIAL_LEADS:
                continue
            segs = _pre_qrs_segments(l, w.beats)
            if len(segs) < 3:
                continue
            c = np.corrcoef(segs)
            correlations.append(np.mean(c[np.triu_indices(len(segs), 1)]))
            examples.append((l.name, segs))
    k = float(np.mean(correlations)) if correlations else float("nan")
    if np.isnan(k):
        verdict = "uncertain"
    elif k > 0.6:
        verdict = "repeatable"
    elif k < 0.35:
        verdict = "absent"
    else:
        verdict = "uncertain"
    return k, verdict, examples


def _atrial_activity(lead, beats):
    """Subtract the average beat shape (P + QRS + T); what remains is atrial activity.
    The template starts 0.30 s before the R peak so a regular P wave is cancelled too and is
    not mistaken for an f wave. Beats cut off at the column edges are cancelled partially.
    Returns the signal and the slice between the first and the last beat's template, the only
    part where every beat is cancelled (outside it a beat that was not detected may remain)."""
    fs = lead.fs
    n0, n1 = int(-0.30 * fs), int(0.45 * fs)
    idx = [int(np.argmin(np.abs(lead.t - tr))) for tr in beats]
    full = [i for i in idx if i + n0 >= 0 and i + n1 <= len(lead.mv)]
    x = lead.mv.copy()
    if len(full) < 2:
        return x, slice(0, 0)
    template = np.mean([x[i + n0:i + n1] for i in full], axis=0)
    for i in idx:
        a, b = max(0, i + n0), min(len(x), i + n1)
        x[a:b] -= template[a - (i + n0):b - (i + n0)]
    return x, slice(max(0, min(idx) + n0), min(len(x), max(idx) + n1))


# common frequency axis, so spectra of scans with different resolutions (fs) can be averaged
SPECTRUM_F = np.arange(0, 20.0001, 0.02)


# Smallest atrial residual (RMS after cancelling the beats) counted as f/F waves. Below it the
# line between beats is flat up to digitization noise; a regular rhythm leaves only a few uV here.
# Very fine fibrillation waves can fall under it - the irregular RR intervals still show then.
F_WAVE_MIN_MV = 0.02


def _atrial_residual(lead, beats):
    """Atrial activity band-passed to 3-12 Hz, and the mask of samples usable for it:
    inside the span where every beat is cancelled and away from each QRS."""
    x, span = _atrial_activity(lead, beats)
    b, a = sps.butter(3, [3, 12], btype="band", fs=lead.fs)
    x = sps.filtfilt(b, a, sps.detrend(x))
    usable = np.zeros(len(x), bool)
    usable[span] = True
    for tr in beats:
        usable &= ~((lead.t > tr - 0.08) & (lead.t < tr + 0.12))
    return x, usable


def _wave_floor(lead):
    """F_WAVE_MIN_MV, or 1.5 px on a coarse scan where one pixel is already a large voltage step."""
    return max(F_WAVE_MIN_MV, 1.5 / lead.px_per_mv)


def atrial_spectrum(windows):
    """Mean spectrum of the atrial activity in the atrial leads.
    Returns (f, p, dominant frequency, organization index, status); status is "ok",
    "none" (the line between beats is flat, no f/F waves) or "no_data" (too few beats)."""
    f = SPECTRUM_F
    band = (f >= 3) & (f <= 12)
    spectra, analyzed = [], 0
    for w in windows:
        for l in w.leads:
            if l.name not in ATRIAL_LEADS:
                continue
            x, span = _atrial_activity(l, w.beats)
            x = x[span]
            if len(x) < 1.5 * l.fs:
                continue
            analyzed += 1
            residual, usable = _atrial_residual(l, w.beats)
            if np.std(residual[usable]) < _wave_floor(l):
                continue
            x = sps.detrend(x)
            b, a = sps.butter(3, [2.5, 15], btype="band", fs=l.fs)
            x = sps.filtfilt(b, a, x)
            nfft = max(8192, len(x))
            p = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
            p = np.interp(f, np.fft.rfftfreq(nfft, 1 / l.fs), p)
            spectra.append(p / p[band].sum())
    if not spectra:
        return None, None, float("nan"), float("nan"), ("none" if analyzed else "no_data")
    p = np.mean(spectra, axis=0)
    fd = float(f[band][np.argmax(p[band])])
    oi = float(p[(f >= fd - 0.5) & (f <= fd + 0.5)].sum() / p[band].sum())
    return f, p, fd, oi, "ok"


def analyze(printouts):
    windows = []
    for pr in printouts:
        for k in sorted({l.column for l in pr.leads}):
            # copies, so trimming does not change the printout and analyze() can run again
            leads = [copy.copy(l) for l in pr.leads if l.column == k]
            for l in leads:
                _trim_edges(l)
            windows.append(Window(name=", ".join(l.name for l in leads), leads=leads))
    for w in windows:
        detect_beats(w)

    rr, rr_window = [], []
    for i, w in enumerate(windows):
        d = np.diff(w.beats)
        rr.extend(d)
        rr_window.extend([i] * len(d))
    rr, rr_window = np.array(rr), np.array(rr_window)
    if rr.size == 0:
        raise NotEnoughBeats()

    # successive differences only within a window (there is a gap in the recording between windows)
    diffs = np.concatenate([np.diff(rr[rr_window == i]) for i in range(len(windows))])
    mean_rr = rr.mean()
    cv = rr.std() / mean_rr
    if diffs.size == 0:
        # at most one interval per segment: nothing to compare, so regularity cannot be judged
        nrmssd, large = float("nan"), float("nan")
    else:
        nrmssd = np.sqrt(np.mean(diffs ** 2)) / mean_rr
        large = float(np.mean(np.abs(diffs) > 0.1 * mean_rr) * 100)
    if np.isnan(nrmssd):
        irregularity = "unknown"
    elif nrmssd > 0.10 and cv > 0.08:
        irregularity = "high"
    elif nrmssd > 0.05:
        irregularity = "moderate"
    else:
        irregularity = "low"

    pk, p_verdict, examples = assess_p_waves(windows)
    f, p, fd, oi, status = atrial_spectrum(windows)
    if status != "ok":
        waves = status
    elif fd >= 5.8 or oi < 0.35:
        waves = "fibrillation"
    elif oi >= 0.5:
        waves = "flutter"
    else:
        waves = "ambiguous"

    result = Result(windows=windows, rr=rr, rr_window=list(rr_window), heart_rate=60 / mean_rr, cv=cv,
                    nrmssd=nrmssd, pct_large_changes=large, irregularity=irregularity,
                    p_correlation=pk, p_waves=p_verdict, spectrum_f=f, spectrum_p=p,
                    dominant_freq=fd, organization_index=oi, atrial_waves=waves, p_examples=examples)
    result.conclusion = _conclusion(result)
    return result


def f_waves(window, lead):
    """Individual atrial (f/F) waves between beats - peaks of the signal after P/QRS/T cancellation."""
    fs = lead.fs
    x, usable = _atrial_residual(lead, window.beats)
    if usable.sum() < fs * 0.5 or np.std(x[usable]) < _wave_floor(lead):
        return np.array([])
    peaks, _ = sps.find_peaks(x, distance=int(0.1 * fs), prominence=0.8 * np.std(x[usable]))
    peaks = [p for p in peaks if usable[p]]
    return lead.t[peaks]


def find_anomalies(result):
    """Concrete places in the recording that deviate from a normal rhythm, with lay explanations."""
    anomalies, waves, p_zones = [], [], []
    mean_rr = result.rr.mean()

    def add(**k):
        k["id"] = f"a{len(anomalies)}"
        anomalies.append(k)

    if result.heart_rate > 100:
        add(type="rate", window=None, t0=None, t1=None, **texts.anomaly_rate(result.heart_rate))

    for i, w in enumerate(result.windows):
        r = w.beats
        rr = np.diff(r)
        for j in range(len(rr)):
            a, b = r[j], r[j + 1]
            if j > 0 and abs(rr[j] - rr[j - 1]) > 0.1 * mean_rr:
                add(type="irregular", window=i, t0=a, t1=b, **texts.anomaly_irregular(rr[j - 1], rr[j]))
            if 60 / rr[j] > 120:
                add(type="short", window=i, t0=a, t1=b, **texts.anomaly_short(rr[j]))
            if rr[j] > 1.5 * mean_rr:
                add(type="pause", window=i, t0=a, t1=b, **texts.anomaly_pause(rr[j], mean_rr))

        # zones before each QRS where a P wave should be
        for tr in r:
            p_zones.append({"window": i, "t0": tr - 0.30, "t1": tr - 0.06})
        if result.p_waves == "absent":
            add(type="noP", window=i, t0=None, t1=None, **texts.anomaly_no_p(i))

        for l in w.leads:
            if l.name not in ATRIAL_LEADS:
                continue
            tf = f_waves(w, l)
            for t in tf:
                k = int(np.argmin(np.abs(l.t - t)))
                waves.append({"window": i, "lead": l.name, "t": float(t), "mv": float(l.mv[k])})
            if len(tf) >= 4:
                rate = 60 / np.median(np.diff(tf))
                add(type="fWaves", window=i, t0=None, t1=None, lead=l.name, **texts.anomaly_f_waves(l.name, rate))
    return anomalies, waves, p_zones


def _conclusion(r):
    if r.irregularity == "high" and r.p_waves != "repeatable":
        return "flutter" if r.atrial_waves == "flutter" else "af"
    if r.irregularity == "low" and r.p_waves == "repeatable":
        return "sinus"
    return "unclear"
