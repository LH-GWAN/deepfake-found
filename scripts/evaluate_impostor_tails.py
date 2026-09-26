"""Measure the impostor tail among faces of one population, against matched controls.

Every identity threshold in this repository was fitted on thirty LFW identities,
of whom at most one is East Asian, while the person most likely to use DeepShield
here is Korean. Face recognisers are known to give more high-scoring impostor
pairs within some demographic groups than within others (NIST FRVT Part 3,
NISTIR 8280), and the impostors that matter to a user are the people who look
like them. This script asks whether that holds for this embedder and whether
the high-confidence threshold keeps its margin inside such a group.

Groups:

east_asian
    LFW identities whose names follow Korean, Chinese or Japanese romanisation,
    by the rules in :func:`population` plus a manual list, excluding the
    evaluation identities. A name is a proxy, not a label; every selected name
    is written to the report so the selection can be audited. Names the rules
    cannot place (a Western given name before one of these family names) are
    ambiguous and kept out of every group, controls included; an identity with
    no photograph the face stack accepts is listed with the reason it was
    dropped.
control_<n>
    For every East Asian identity, another LFW identity with at least as many
    photographs, the same predicted sex and a predicted age within ten years,
    drawn without replacement and represented by the same number of photographs.
    Three independent draws by default. Matching sex and age separates a
    population effect from the East Asian set being mostly older men, which
    LFW's press photographs make it.

Within each group every accepted photograph is a probe, clean and as
``video_small`` (0.55 scale, H.264 crf 35), and identities with at least two
accepted photographs form galleries of clean photographs. A probe's impostor
score against an identity is its maximum similarity over that identity's
gallery, as the matcher's ``max`` aggregation computes it. Pairs of names whose
clean photographs score above 0.5 are dropped as the same person filed twice.
Genuine scores are leave-one-out. Sex and age come from InsightFace's
``genderage`` model in the ``buffalo_l`` pack, so nothing is downloaded.

Usage:
    python scripts/evaluate_impostor_tails.py
    python scripts/evaluate_impostor_tails.py --controls 1 --workers 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import load_config
from deepshield.experiments import environment
from deepshield.media import load_image
from deepshield.transforms import Transformation

KOREAN = (
    "kim lee park choi roh moon yoon chung jung kang cho han hwang ahn song hong shin kwon oh "
    "seo yoo jang lim yim ko nam baek chang jeong jeon bae yun seong suh sohn son choe pak paek "
    "ri rhee heo noh ryu yu ha kwak jo yeo won min pyo bak cha goh suk chyung ham"
).split()
CHINESE = (
    "hu jiang wen zhu li wang zhang chen liu yang huang zhao wu zhou xu sun ma lin tang deng "
    "xie feng cao qian jiao luo gao liang tsai lu lui yeo goh chok soong guo he xi ye zeng peng "
    "pan dong yuan su cai tian du qin ren shen xiong jin wei yin yan kong hao fang shi bai ding "
    "wan zou meng qiu gu yi lou hung tung hsu hsieh chiang chu kuo tsao lo wong chan cheung "
    "leung lau ho chow kwok tsang yip fong lam tse lai chiu kwan yeoh ng tan jia gong xiang "
    "chao ling"
).split()
JAPANESE = (
    "koizumi tanaka suzuki takahashi watanabe ito yamamoto nakamura kobayashi kato yoshida "
    "yamada sasaki matsumoto inoue kimura hayashi shimizu mori abe ikeda hashimoto ishikawa "
    "ogawa okada fujita goto murakami sakamoto saito fukuda nishimura fujii kaneko okamoto "
    "fujiwara miura nakajima ishii ueda morita harada sakai miyazaki ishida takeda murata ueno "
    "masuda hirano matsui noguchi nomura kikuchi sugimoto arai hamada ichikawa mizuno yamashita "
    "ishihara otani nakasone obuchi kawaguchi tanigaki hiranuma machimura ozawa kan hatoyama aso "
    "sato ichiro nomo ohno takako sugiyama morigami tamura mitarai urushima hasuike takenaka "
    "naemura nakata gomi oguchi owada soga oshitani inamoto tabei kutaragi haraguchi matsuura "
    "uehara kitajima nakayama nagasawa tokuyama azuma hayami koshiba moriyama yabunaka "
    "kawabuchi nagashima ishiba katayama suetsugu taniguchi sorimachi kanzaki fukui kitano "
    "yamasaki fuji masumoto maeda hagiwara takagi izawa motegi takebe akashi chimura kamei "
    "okudo miyazato fujimori hosoi ohata idei yoshino"
).split()
SURNAMES = frozenset(KOREAN + CHINESE)
# One syllable of Chinese (pinyin, Wade-Giles, Cantonese) or Korean romanisation.
# Deliberately loose: it only has to tell "Changchun" or "Myung" from "Clijsters".
SYLLABLE = (
    r"(?:ch|sh|zh|ts|tz|hs|hw|wh|kw|gw|jj|kk|tt|pp|ss|[bpmfdtnlgkhjqxrzcsyw])?"
    r"(?:iao|iu|ia|ie|io|ua|uo|ue|ui|ai|ao|ay|ei|eo|eu|ey|ou|oo|ee|oe|ae|oi|yu|ye|ya|yi|yo"
    r"|[aeiou])"
    r"(?:ng|ck|n|m|k|p|t|l|h|r)?"
)
ROMANISED = re.compile(rf"(?:{SYLLABLE}){{1,3}}")
# One mora of Hepburn romanisation, including the doubled consonant and a closing n.
MORAE = re.compile(
    r"(?:(?:(?:[kgsztdnhbpmr]y?|sh|ch|ts|j|f|y|w)?[aiueo])|n(?![aiueoy])|([kstp])(?=\1))+"
)
MANUAL = {
    "Zhang_Ziyi", "Gong_Li", "Yao_Ming", "Jackie_Chan", "Lucy_Liu", "Michelle_Kwan",
    "Chen_Kaige", "Tang_Jiaxuan", "Qian_Qichen", "Chen_Liang_Yu", "Maggie_Cheung",
    "Zhong_Nanshan", "Annette_Lu", "Tung_Chee-hwa", "Toshihiko_Fukui", "Yoko_Ono",
    "Hiroyuki_Yoshino", "Nobuyuki_Idei", "Wang_Yingfan", "Nan_Wang", "Zhang_Wenkang",
    "Liu_Mingkang", "Li_Peng",
    # Western or English given names before a Chinese, Korean or Japanese family name,
    # which the rules leave ambiguous: public figures of East Asian descent.
    "Ambrose_Lee", "Andy_Lau", "Antony_Leung", "Bill_Kong", "Cecilia_Cheung", "Connie_Chung",
    "David_Ho", "Edward_Lu", "Elaine_Chao", "Fann_Wong", "Faye_Wong", "Frank_Hsieh",
    "Fruit_Chan", "Jacky_Cheung", "Kurt_Suzuki", "Leon_Lai", "Lisa_Ling", "Michael_Chang",
    "Michelle_Yeoh", "Nicholas_Tse", "Peter_Chan",
    # Given name first, written as one pinyin or Korean word.
    "Chuanyun_Li", "Xiang_Xu", "Yingfan_Wang", "Yishan_Zhang", "Ziwang_Xu", "Myung_Yang",
    "Soon_Yi", "Fujio_Cho", "Moon-So-ri", "Rod_Jong-il", "Guangdong_Ou_Guangyuan",
    "Debra_Yang",
}
# Names the rules read as East Asian that the selection could not confirm.
AMBIGUOUS = {"Lou_Lang"}
# Names the rules read as romanised that belong to people who are not East Asian.
NOT_EAST_ASIAN = {
    "Chan_Gailey", "Duane_Lee_Chapman", "Jo_Dee_Messina", "Lee_Baca", "Lou_Piniella",
    "Mary_Lou_Retton", "Paul_Lo_Duca", "Spike_Lee", "Ben_Lee",
}
LONGEST_GIVEN_PART = 6


def _romanised(token: str, longest: int = 12) -> bool:
    """Return whether a name token reads as romanised Chinese or Korean syllables."""
    pieces = token.split("-")
    return len(pieces) <= 3 and all(
        piece and len(piece) <= longest and ROMANISED.fullmatch(piece) for piece in pieces
    )


def population(name: str) -> str:
    """Return ``east_asian``, ``ambiguous`` or ``other`` for an LFW name.

    ``east_asian``: a Korean or Chinese family name followed by a romanised
    given name (``Li_Changchun``, ``Kim_Dae-jung``, ``Chan_Ho_Park`` read either
    way round when the given name is hyphenated or in two parts), an English
    name before a Chinese one (``Alan_Tang_Kwong-wing``), a Japanese family name
    after a given name written in Hepburn morae, or the manual list.
    ``ambiguous``: a family name from these lists after a given name the rules
    cannot read (``Michelle_Yeoh``, ``Spike_Lee``, ``Kurt_Suzuki``). Such people
    join neither group: a control must not hold East Asian faces either, or the
    contrast it exists for shrinks. A name is a proxy, not a label; the report
    lists every name in every group, and the ambiguous ones, for audit.
    """
    if name in MANUAL:
        return "east_asian"
    if name in NOT_EAST_ASIAN:
        return "other"
    if name in AMBIGUOUS:
        return "ambiguous"
    parts = name.lower().split("_")
    first, last = parts[0], parts[-1]
    # Three-part names are two short given syllables; longer tokens are Western.
    longest = 12 if len(parts) == 2 else LONGEST_GIVEN_PART
    if len(parts) in (2, 3):
        if first in SURNAMES and all(_romanised(part, longest) for part in parts[1:]):
            return "east_asian"
        given_first = len(parts) == 3 or "-" in first
        if last in SURNAMES and given_first and all(
            _romanised(part, longest) for part in parts[:-1]
        ):
            return "east_asian"
        if last in JAPANESE and len(parts) == 2 and MORAE.fullmatch(first):
            return "east_asian"
    if len(parts) == 3 and parts[1] in SURNAMES and _romanised(last, LONGEST_GIVEN_PART) and (
        "-" in last or len(last) <= 4
    ):
        # An English name before a Chinese one: Alan_Tang_Kwong-wing, Vicki_Zhao_Wei.
        return "east_asian"
    if last in SURNAMES or last in JAPANESE or name in AMBIGUOUS:
        return "ambiguous"
    return "other"


def east_asian(name: str) -> bool:
    """Return whether :func:`population` places an LFW name in the East Asian group."""
    return population(name) == "east_asian"


SMALL = Transformation("video_small", "video_compression", {"scale": 0.55, "crf": 35})
CONDITIONS = ("clean", "video_small")
PHOTOS_PER_IDENTITY = 8
MIN_FACE_PIXELS = 60
MIN_DETECTION_CONFIDENCE = 0.7
ALIAS_SIMILARITY = 0.5
AGE_TOLERANCE = 10.0
LEVELS = (0.25, 0.30, 0.35)

_components: tuple[Any, Any, Any] | None = None


def identities(lfw: Path, excluded: set[str]) -> dict[str, list[Path]]:
    """Return every LFW identity's photographs, leaving out the excluded names."""
    return {
        folder.name: sorted(folder.glob("*.jpg"))
        for folder in sorted(lfw.iterdir())
        if folder.is_dir() and folder.name.lower() not in excluded
    }


def evaluation_identities(faces: Path) -> set[str]:
    """Return the evaluation set's identity names, lower-cased as LFW folder names."""
    listing = faces / "identities.txt"
    if not listing.is_file():
        return set()
    return {
        line.split("\t")[1].strip()
        for line in listing.read_text(encoding="utf-8").splitlines()
        if "\t" in line
    }


class Attributes:
    """Predicted sex and age of an identity, from its first photograph, cached."""

    def __init__(self, models: Path) -> None:
        """Load InsightFace's detector and ``genderage`` model from the local pack."""
        import insightface

        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=str(models),
            allowed_modules=["detection", "genderage"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.cache: dict[str, tuple[str, float] | None] = {}

    def __call__(self, name: str, photo: Path) -> tuple[str, float] | None:
        """Return ``(sex, age)`` for the largest face in ``photo``, or ``None``."""
        if name not in self.cache:
            image = load_image(photo)
            faces = self.app.get(np.ascontiguousarray(image[:, :, ::-1]))
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                self.cache[name] = (str(face.sex), float(face.age))
            else:
                self.cache[name] = None
        return self.cache[name]


def matched_controls(
    group: list[tuple[str, int]],
    pool: list[str],
    photos: dict[str, list[Path]],
    attributes: Attributes,
    rng: np.random.Generator,
    taken: set[str],
) -> tuple[list[tuple[str, int]], int]:
    """Draw one control per group identity; return them and how many fell back to count only."""
    controls: list[tuple[str, int]] = []
    fallbacks = 0
    for name, need in group:
        target = attributes(name, photos[name][0])
        candidates = [
            other for other in pool if other not in taken and len(photos[other]) >= need
        ]
        order = rng.permutation(len(candidates))
        chosen = None
        for index in order[:400]:
            other = candidates[index]
            found = attributes(other, photos[other][0])
            if target is None or found is None:
                continue
            if found[0] == target[0] and abs(found[1] - target[1]) <= AGE_TOLERANCE:
                chosen = other
                break
        if chosen is None:
            chosen = candidates[order[0]]
            fallbacks += 1
        taken.add(chosen)
        controls.append((chosen, need))
    return controls, fallbacks


def _embed(paths: list[str]) -> list[dict[str, Any]]:
    """Embed each photograph clean and as ``video_small`` with the configured face stack."""
    global _components
    if _components is None:
        from deepshield.face.aligner import build_aligner
        from deepshield.face.detector import build_detector
        from deepshield.face.embedder import build_embedder

        config = load_config()
        _components = (
            build_detector(config.face.detector),
            build_aligner(config.face.aligner),
            build_embedder(config.face.embedder),
        )
    detector, aligner, embedder = _components
    rows: list[dict[str, Any]] = []
    for path in paths:
        image = load_image(Path(path))
        row: dict[str, Any] = {"path": path, "accepted": False}
        faces = detector.detect(image)
        if len(faces) != 1:
            row["rejected"] = f"{len(faces)} faces found"
        elif faces[0].detection_confidence < MIN_DETECTION_CONFIDENCE:
            row["rejected"] = "detection confidence below 0.7"
        elif min(faces[0].bbox.width, faces[0].bbox.height) < MIN_FACE_PIXELS:
            row["rejected"] = f"face smaller than {MIN_FACE_PIXELS} pixels"
        else:
            row["clean"] = embedder.embed(aligner.align(image, faces[0]).image).vector
            small = SMALL.apply(image)
            found = detector.detect(small)
            if found:
                best = max(found, key=lambda f: f.detection_confidence)
                row["video_small"] = embedder.embed(aligner.align(small, best).image).vector
                row["accepted"] = True
            else:
                row["rejected"] = "no face after video_small"
        rows.append(row)
    return rows


def embed_all(paths: list[Path], workers: int) -> dict[str, dict[str, Any]]:
    """Embed every photograph, split across ``workers`` processes."""
    chunks = [[str(p) for p in paths[start::workers]] for start in range(workers)]
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(_embed, chunks):
            for row in rows:
                results[row["path"]] = row
    return results


def tail_estimate(scores: np.ndarray, level: float) -> float | None:
    """Estimate ``P(score >= level)`` from a generalised Pareto fit above the 99th percentile.

    The fit extrapolates beyond the observed maximum, so it is a rough guide to
    how quickly the tail thins, not a measurement.
    """
    from scipy import stats

    threshold = float(np.percentile(scores, 99))
    excess = scores[scores > threshold] - threshold
    if excess.size < 20:
        return None
    shape, _, scale = stats.genpareto.fit(excess, floc=0.0)
    return float(
        excess.size / scores.size * stats.genpareto.sf(level - threshold, shape, 0.0, scale)
    )


def analyse(
    members: Iterable[tuple[str, int]],
    photos: dict[str, list[Path]],
    embedded: dict[str, dict[str, Any]],
    high: float,
) -> dict[str, Any]:
    """Score every probe against every other identity's gallery within one group."""
    names: list[str] = []
    clean: list[np.ndarray] = []
    small: list[np.ndarray] = []
    dropped: dict[str, list[str]] = {}
    for name, count in members:
        reasons: list[str] = []
        for path in photos[name][:count]:
            entry = embedded.get(str(path))
            if entry and entry["accepted"]:
                names.append(name)
                clean.append(np.asarray(entry["clean"], dtype=np.float64))
                small.append(np.asarray(entry["video_small"], dtype=np.float64))
            else:
                reasons.append(entry.get("rejected", "not embedded") if entry else "not embedded")
        if name not in names:
            dropped[name] = sorted(set(reasons))
    labels = np.asarray(names)
    gallery = np.stack(clean)
    counts = {name: int((labels == name).sum()) for name in set(names)}
    galleries = sorted(name for name, count in counts.items() if count >= 2)
    same_person = (gallery @ gallery.T > ALIAS_SIMILARITY) & (labels[:, None] != labels[None, :])
    aliases = sorted(
        {tuple(sorted((labels[i], labels[j]))) for i, j in np.argwhere(same_person)}
    )

    report: dict[str, Any] = {
        "identities": len(counts),
        "photos": len(names),
        "gallery_identities": len(galleries),
        "measured_members": sorted(counts),
        "dropped_no_usable_photo": dropped,
        "dropped_as_same_person": [list(pair) for pair in aliases],
    }
    for condition, probes in (("clean", gallery), ("video_small", np.stack(small))):
        similarity = probes @ gallery.T
        impostor: list[float] = []
        pairs: list[tuple[float, str, str]] = []
        genuine: list[float] = []
        for identity in galleries:
            columns = np.where(labels == identity)[0]
            for row in range(len(labels)):
                if labels[row] == identity:
                    others = columns[columns != row]
                    genuine.append(float(similarity[row, others].max()))
                elif not same_person[row, columns].any():
                    score = float(similarity[row, columns].max())
                    impostor.append(score)
                    pairs.append((score, str(labels[row]), identity))
        scores = np.asarray(impostor)
        pairs.sort(reverse=True)
        report[condition] = {
            "impostor_comparisons": int(scores.size),
            "impostor_max": round(float(scores.max()), 4),
            "impostor_p99": round(float(np.percentile(scores, 99)), 4),
            "impostor_p999": round(float(np.percentile(scores, 99.9)), 4),
            "impostor_at_or_above": {
                f"{level:.2f}": int((scores >= level).sum()) for level in LEVELS
            },
            "impostor_at_or_above_high_confidence": int((scores >= high).sum()),
            "identity_pairs_at_or_above_0.25": len(
                {tuple(sorted(pair[1:])) for pair in pairs if pair[0] >= 0.25}
            ),
            "tail_estimate_at_high_confidence": tail_estimate(scores, high),
            "top_pairs": [
                {"similarity": round(score, 4), "probe": probe, "gallery": gallery_name}
                for score, probe, gallery_name in pairs[:6]
            ],
            "genuine_comparisons": len(genuine),
            "genuine_at_or_above_high_confidence": round(
                float(np.mean(np.asarray(genuine) >= high)), 4
            ),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    """Select the groups, embed their photographs and compare their impostor tails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lfw", type=Path, default=Path("data/sklearn/lfw_home/lfw_funneled")
    )
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--models", type=Path, default=Path("models/insightface"))
    parser.add_argument("--controls", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    if not args.lfw.is_dir():
        raise SystemExit(f"{args.lfw} not found; run scripts/build_evaluation_set.py first")
    photos = identities(args.lfw, evaluation_identities(args.faces))
    group = [
        (name, min(len(photos[name]), PHOTOS_PER_IDENTITY))
        for name in sorted(photos)
        if east_asian(name)
    ]
    ambiguous = sorted(name for name in photos if population(name) == "ambiguous")
    # Controls come only from names read as neither East Asian nor ambiguous.
    pool = sorted(name for name in photos if population(name) == "other")
    attributes = Attributes(args.models)
    rng = np.random.default_rng(args.seed)

    groups: dict[str, list[tuple[str, int]]] = {"east_asian": group}
    fallbacks: dict[str, int] = {}
    taken: set[str] = set()
    for draw in range(args.controls):
        controls, missed = matched_controls(group, pool, photos, attributes, rng, taken)
        groups[f"control_{draw}"] = controls
        fallbacks[f"control_{draw}"] = missed
        print(f"control_{draw}: {len(controls)} identities, {missed} matched on count only")

    wanted = sorted(
        {path for members in groups.values() for name, count in members
         for path in photos[name][:count]}
    )
    print(f"embedding {len(wanted)} photographs clean and as video_small", flush=True)
    embedded = embed_all(wanted, max(1, args.workers))

    high = load_config().thresholds.face_similarity.high_confidence_threshold
    report: dict[str, Any] = {
        "question": "is the impostor tail heavier within one population than in matched controls?",
        "high_confidence_threshold": high,
        "selection": {
            "east_asian_selected": len(group),
            "ambiguous_excluded_from_every_group": ambiguous,
        },
        "groups": {},
        # The seed that drew the controls, not the configuration's default.
        "environment": {**environment(load_config()), "random_seed": args.seed},
    }
    for label, members in groups.items():
        result = analyse(members, photos, embedded, high)
        # Describe the identities that were measured, not every one selected.
        described = [attributes(name, photos[name][0]) for name in result["measured_members"]]
        known = [item for item in described if item is not None]
        result["members"] = [name for name, _ in members]
        result["predicted_male_share"] = round(
            float(np.mean([sex == "M" for sex, _ in known])), 3
        ) if known else None
        result["predicted_median_age"] = (
            round(float(np.median([age for _, age in known])), 1) if known else None
        )
        result["matched_on_count_only"] = fallbacks.get(label, 0)
        report["groups"][label] = result

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "impostor_tails.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'group':12s} {'condition':12s} {'n':>6s} {'max':>6s} {'p99.9':>6s} "
          f"{'>=0.25':>7s} {'>=high':>7s}  male  age")
    for label, result in report["groups"].items():
        for condition in CONDITIONS:
            row = result[condition]
            above = row["impostor_at_or_above"]
            print(
                f"{label:12s} {condition:12s} {row['impostor_comparisons']:>6d} "
                f"{row['impostor_max']:>6.3f} {row['impostor_p999']:>6.3f} "
                f"{above['0.25']:>7d} {row['impostor_at_or_above_high_confidence']:>7d}  "
                f"{result['predicted_male_share']}  {result['predicted_median_age']}"
            )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
