"""DeepShield command line interface.

Every command reports what was measured and what that measurement does not
establish. Commands that cannot run because a model or an enrolled identity is
missing say which, and exit with a distinct code so scripts can tell a
configuration problem from a finding.

Exit codes:
    0  success
    1  runtime error
    2  usage error
    3  command recognised but not implemented in the current phase
    4  a required model or dependency is unavailable
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepshield import __version__
from deepshield.config import DeepShieldConfig, load_config
from deepshield.detection.deepfake import DEEPFAKE_REGISTRY
from deepshield.exceptions import (
    DeepShieldError,
    EnrollmentError,
    InvalidMediaError,
    ModelNotAvailableError,
    NotImplementedInPhaseError,
)
from deepshield.face.aligner import ALIGNER_REGISTRY
from deepshield.face.detector import DETECTOR_REGISTRY
from deepshield.face.embedder import EMBEDDER_REGISTRY
from deepshield.logging_utils import configure_logging, get_logger
from deepshield.protection.watermark import WATERMARK_REGISTRY
from deepshield.registry import ComponentRegistry

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_IMPLEMENTED = 3
EXIT_MODEL_UNAVAILABLE = 4

PHASE_OWNERS: dict[str, str] = {}

REGISTRIES: dict[str, ComponentRegistry[Any]] = {
    "face_detector": DETECTOR_REGISTRY,
    "face_aligner": ALIGNER_REGISTRY,
    "face_embedder": EMBEDDER_REGISTRY,
    "deepfake_detector": DEEPFAKE_REGISTRY,
    "watermark": WATERMARK_REGISTRY,
}

IMPLEMENTED_COMMANDS = (
    "info",
    "doctor",
    "download-models",
    "enroll",
    "identities",
    "protect",
    "analyze-image",
    "analyze-video",
    "watermark-detect",
    "fingerprint",
    "provenance",
    "report",
    "robustness-test",
    "serve",
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="deepshield",
        description=(
            "Personal deepfake identity protection and multi-signal detection system. "
            "This tool reports identity similarity and synthetic-media likelihood; "
            "it cannot prove that an image was used as training data."
        ),
    )
    parser.add_argument("--version", action="version", version=f"deepshield {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="path to default.yaml")
    parser.add_argument("--thresholds", type=Path, default=None, help="path to thresholds.yaml")
    parser.add_argument("--log-level", default=None, help="override the configured log level")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    sub = parser.add_subparsers(dest="command", metavar="command")

    sub.add_parser("info", help="show configuration, phase status and registered backends")
    sub.add_parser("doctor", help="check the environment and report missing optional deps")

    p_models = sub.add_parser("download-models", help="fetch model weights")
    p_models.add_argument(
        "--skip-insightface",
        action="store_true",
        help="skip the ArcFace pack needed by the default embedder",
    )

    p_enroll = sub.add_parser("enroll", help="enroll a user identity from reference images")
    p_enroll.add_argument("images_dir", type=Path, help="directory of reference face images")
    p_enroll.add_argument("--user-id", required=True, help="identifier for the enrolled user")
    p_enroll.add_argument(
        "--min-images",
        type=int,
        default=None,
        help=(
            "lower the minimum number of usable reference photos. Fewer photos means "
            "less pose and lighting variation, so similarity scores get less reliable"
        ),
    )

    sub.add_parser("identities", help="list enrolled identities")

    p_protect = sub.add_parser("protect", help="apply the protection pipeline to one image")
    p_protect.add_argument("image", type=Path)
    p_protect.add_argument("--user-id", required=True)
    p_protect.add_argument("--distribution-id", default=None)
    p_protect.add_argument("--output", type=Path, default=None)

    p_img = sub.add_parser("analyze-image", help="analyse one suspect image")
    p_img.add_argument("image", type=Path)
    p_img.add_argument("--user-id", default=None)
    p_img.add_argument("--save", action="store_true", help="store the evidence record")

    p_vid = sub.add_parser("analyze-video", help="analyse one suspect video")
    p_vid.add_argument("video", type=Path)
    p_vid.add_argument("--user-id", default=None)
    p_vid.add_argument("--save", action="store_true", help="store the evidence record")

    p_wm = sub.add_parser("watermark-detect", help="attempt watermark extraction")
    p_wm.add_argument("image", type=Path)

    p_fp = sub.add_parser("fingerprint", help="compute the fingerprints of one image")
    p_fp.add_argument("image", type=Path)

    p_prov = sub.add_parser("provenance", help="show the lineage of a registered asset")
    p_prov.add_argument("asset_id")

    p_rb = sub.add_parser("robustness-test", help="run a transformation robustness benchmark")
    p_rb.add_argument("images", type=Path, nargs="+", help="image files or a directory")
    p_rb.add_argument(
        "--experiment",
        default="face",
        choices=["face", "watermark", "combined"],
        help="which benchmark to run",
    )
    p_rb.add_argument("--output", type=Path, default=None)

    p_rep = sub.add_parser("report", help="render a stored evidence record")
    p_rep.add_argument("analysis_id")

    p_serve = sub.add_parser("serve", help="run the REST API")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)

    return parser


def _emit(payload: dict[str, Any], as_json: bool, lines: Sequence[str]) -> None:
    """Print either the JSON payload or the human-readable lines."""
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    for line in lines:
        print(line)


def command_info(config: DeepShieldConfig, as_json: bool) -> int:
    """Report the active configuration, backends and calibration status."""
    backends: dict[str, list[str]] = {
        kind: registry.available() for kind, registry in REGISTRIES.items()
    }
    selected = {
        "face_detector": config.face.detector.backend,
        "face_aligner": config.face.aligner.backend,
        "face_embedder": config.face.embedder.backend,
        "deepfake_detector": config.detection.deepfake.backend,
        "watermark": config.protection.watermark.backend,
    }
    face_thresholds = config.thresholds.face_similarity
    payload = {
        "version": __version__,
        "device": config.runtime.device,
        "random_seed": config.runtime.random_seed,
        "selected_backends": selected,
        "registered_backends": backends,
        "thresholds_calibrated": {
            "face_similarity": face_thresholds.calibrated,
            "deepfake": config.thresholds.deepfake.calibrated,
        },
        "face_thresholds": {
            "candidate": face_thresholds.candidate_threshold,
            "high_confidence": face_thresholds.high_confidence_threshold,
            "source": face_thresholds.calibration_source,
        },
        "commands": {"implemented": list(IMPLEMENTED_COMMANDS), "planned": PHASE_OWNERS},
    }

    deepfake_note = (
        "calibrated"
        if config.thresholds.deepfake.calibrated
        else "NOT calibrated; excluded from the risk score"
    )
    calibration_note = (
        f"calibrated from {face_thresholds.calibration_source}"
        if face_thresholds.calibrated
        else "NOT calibrated; treat as provisional"
    )
    lines = [
        f"deepshield {__version__}",
        f"device: {config.runtime.device}   seed: {config.runtime.random_seed}",
        "",
        "selected backends:",
        *(f"  {k:<18} {v}" for k, v in selected.items()),
        "",
        "registered backends:",
        *(f"  {k:<18} {', '.join(v)}" for k, v in backends.items()),
        "",
        "thresholds:",
        f"  face candidate       {face_thresholds.candidate_threshold:.4f}",
        f"  face high confidence {face_thresholds.high_confidence_threshold:.4f}",
        f"  face similarity      {calibration_note}",
        f"  deepfake             {deepfake_note}",
        "",
        f"commands: {', '.join(IMPLEMENTED_COMMANDS)}",
    ]
    _emit(payload, as_json, lines)
    return EXIT_OK


def command_doctor(config: DeepShieldConfig, as_json: bool) -> int:
    """Report which optional dependencies and model weights are present."""
    from deepshield.models import available_models

    optional = {
        "cv2": "face detection, alignment and video decoding",
        "onnxruntime": "ONNX model execution",
        "insightface": "SCRFD detection and ArcFace embeddings",
        "fastapi": "REST API",
        "uvicorn": "REST API server",
        "sklearn": "extended metrics",
        "pandas": "experiment tables",
    }
    status: dict[str, dict[str, Any]] = {}
    for module, purpose in optional.items():
        try:
            __import__(module)
            available = True
        except ImportError:
            available = False
        status[module] = {"available": available, "purpose": purpose}

    models = available_models(config.runtime.model_dir)
    payload = {
        "python": sys.version.split()[0],
        "optional_dependencies": status,
        "models": models,
        "model_dir": str(config.runtime.model_dir),
    }
    lines = [
        f"python {payload['python']}",
        "",
        "optional dependencies:",
        *(
            f"  [{'x' if info['available'] else ' '}] {name:<14} {info['purpose']}"
            for name, info in status.items()
        ),
        "",
        f"model weights in {config.runtime.model_dir}:",
        *(
            f"  [{'x' if info['present'] else ' '}] {name:<14} {info['description']}"
            for name, info in models.items()
        ),
    ]
    _emit(payload, as_json, lines)
    return EXIT_OK


def command_download_models(config: DeepShieldConfig, args: Any, as_json: bool) -> int:
    """Fetch the model weights the configured backends need."""
    from deepshield.models import MODEL_ASSETS, download_asset

    fetched = {}
    for key, asset in MODEL_ASSETS.items():
        path = download_asset(asset, config.runtime.model_dir)
        fetched[key] = str(path)
        if not as_json:
            print(f"ready {key:<8} {path}")

    if not args.skip_insightface:
        from deepshield.config import FaceEmbedderConfig
        from deepshield.face.backends import InsightFaceEmbedder

        embedder = InsightFaceEmbedder(
            FaceEmbedderConfig(backend="insightface"), model_dir=config.runtime.model_dir
        )
        fetched["insightface"] = str(embedder.model_path)
        if not as_json:
            print(f"ready insightface {embedder.model_path}")

    _emit({"models": fetched}, as_json, [])
    return EXIT_OK


def command_enroll(config: DeepShieldConfig, args: Any, as_json: bool) -> int:
    """Enroll a user identity from a directory of reference photographs.

    The directory is checked before the enroller is built, because building one
    loads face models and may download them. A mistyped path must be reported as
    a missing directory, not as a failed download.
    """
    from deepshield.face.enrollment import DefaultIdentityEnroller
    from deepshield.storage import build_identity_repository

    settings = config
    if args.min_images is not None:
        settings = config.model_copy(
            update={
                "enrollment": config.enrollment.model_copy(
                    update={"min_images": args.min_images}
                )
            }
        )
        if args.min_images < config.enrollment.min_images:
            print(
                f"warning: minimum reference images lowered to {args.min_images} "
                f"(default {config.enrollment.min_images}). Fewer photos capture less "
                "pose and lighting variation, so similarity scores become less reliable.",
                file=sys.stderr,
            )

    if not Path(args.images_dir).is_dir():
        raise EnrollmentError(f"enrollment directory not found: {args.images_dir}")

    enroller = DefaultIdentityEnroller(settings)
    result = enroller.enroll_directory(args.user_id, args.images_dir)
    build_identity_repository(settings).save(result.profile)

    payload = result.to_dict()
    lines = [
        f"enrolled '{args.user_id}' from {result.accepted_count} of {len(result.reports)} images",
        f"  model      {result.profile.model.name} ({result.profile.embedding_dimension}d)",
        "",
        "images:",
        *(
            f"  [{'x' if r.accepted else ' '}] {Path(r.path).name:<24} {r.reason}"
            for r in result.reports
        ),
        "",
        "Embeddings are biometric data and are stored separately from image files.",
    ]
    _emit(payload, as_json, lines)
    return EXIT_OK


def command_identities(config: DeepShieldConfig, as_json: bool) -> int:
    """List every enrolled identity."""
    from deepshield.storage import build_identity_repository

    repository = build_identity_repository(config)
    profiles = repository.load_all()
    payload = {"identities": [p.to_dict() for p in profiles]}
    lines = (
        [
            f"{p.user_id:<20} {p.image_count} references  "
            f"{p.model.name} ({p.embedding_dimension}d)  enrolled {p.created_at[:19]}"
            for p in profiles
        ]
        if profiles
        else ["no identities enrolled"]
    )
    _emit(payload, as_json, lines)
    return EXIT_OK


def command_protect(config: DeepShieldConfig, args: Any, as_json: bool) -> int:
    """Protect one image and register it as an asset."""
    from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline

    report = DefaultProtectionPipeline(config).protect(
        args.image, args.user_id, args.distribution_id, args.output
    )
    watermark = report["watermark"]
    lines = [
        f"protected {report['source_path']}",
        f"  output      {report['protected_path']}",
        f"  asset id    {report['asset_id']}",
        f"  watermark   {watermark['code']} ({watermark['backend']}), "
        f"verified after save: {watermark['verified_after_save']}",
        f"  quality     PSNR {report['quality']['psnr']} dB, SSIM {report['quality']['ssim']}",
        f"  sha256      {report['fingerprint']['sha256'][:32]}...",
        "",
        *(f"  note: {line}" for line in report["limitations"]),
    ]
    _emit(report, as_json, lines)
    return EXIT_OK


def _render_evidence(payload: dict[str, Any]) -> list[str]:
    """Render an evidence record as an explainable text report."""
    identity = payload["identity"]
    risk = payload.get("risk") or {}
    lines = [
        f"analysis of {payload['source_id']}  ({payload['media']['type']})",
        f"  sha256   {(payload['media'].get('sha256') or '')[:32]}...",
        "",
        payload.get("summary", ""),
        "",
        "evidence:",
        f"  face similarity     {identity['similarity']}"
        + (f"  matched: {identity['matched_user_id']}" if identity["matched_user_id"] else ""),
        f"  synthetic score     {payload['deepfake']['score']}",
        f"  watermark           detected={payload['watermark']['detected']} "
        f"confidence={payload['watermark']['confidence']} code={payload['watermark']['code']}",
        f"  fingerprint         {payload['fingerprint']['perceptual_similarity']}"
        + (
            f"  asset: {payload['fingerprint']['matched_asset_id']}"
            if payload["fingerprint"].get("matched_asset_id")
            else ""
        ),
        f"  provenance          {payload['provenance']['confidence']}",
        "",
    ]
    if risk:
        lines += [
            f"risk score: {risk['risk_score']} / 100  ({risk['risk_level']})",
            *(f"  {line}" for line in risk.get("explanation", [])),
            "",
        ]
    lines += [
        "limitations:",
        *(f"  - {line}" for line in payload.get("limitations", [])),
    ]
    if payload.get("processing_seconds") is not None:
        lines += ["", f"processed in {payload['processing_seconds']}s"]
    return lines


def command_analyze(config: DeepShieldConfig, args: Any, as_json: bool, video: bool) -> int:
    """Analyse one image or video and print its evidence record.

    The input file is checked before the pipeline is built, for the same reason
    as in :func:`command_enroll`.
    """
    from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline
    from deepshield.storage import build_evidence_repository

    source = args.video if video else args.image
    if not Path(source).is_file():
        raise InvalidMediaError(f"{'video' if video else 'image'} not found: {source}")

    pipeline = DefaultAnalysisPipeline(config)
    record = (
        pipeline.analyze_video(args.video, args.user_id)
        if video
        else pipeline.analyze_image(args.image, args.user_id)
    )
    if args.save:
        record.analysis_id = build_evidence_repository(config).save(record)

    payload = record.to_dict()
    lines = _render_evidence(payload)
    if record.analysis_id:
        lines += ["", f"stored as analysis {record.analysis_id}"]
    _emit(payload, as_json, lines)
    return EXIT_OK


def command_watermark_detect(config: DeepShieldConfig, args: Any, as_json: bool) -> int:
    """Attempt watermark extraction from one image."""
    from deepshield.media import load_image
    from deepshield.protection.watermark import build_watermarker
    from deepshield.storage import build_asset_repository

    result = build_watermarker(config.protection.watermark).detect(load_image(args.image))
    asset = (
        build_asset_repository(config).find_by_watermark_code(result.watermark_code)
        if result.watermark_code
        else None
    )
    payload = result.to_dict()
    payload["matched_asset"] = asset.to_dict() if asset else None
    lines = [
        f"watermark detection for {args.image}",
        f"  detected    {result.detected}",
        f"  confidence  {result.confidence:.4f}",
        f"  code        {result.watermark_code}",
        f"  backend     {result.backend}",
    ]
    if asset:
        lines += [
            f"  asset       {asset.asset_id} (user {asset.user_id})",
            f"  channel     {asset.distribution_id}",
        ]
    lines += [
        "",
        "A negative result is inconclusive: cropping, rescaling and regeneration all",
        "remove the mark. Only a positive detection is evidence.",
    ]
    _emit(payload, as_json, lines)
    return EXIT_OK


def command_fingerprint(config: DeepShieldConfig, args: Any, as_json: bool) -> int:
    """Compute and compare the fingerprints of one image."""
    from deepshield.protection.fingerprint import DefaultFingerprinter, hash_similarity
    from deepshield.storage import build_asset_repository

    fingerprinter = DefaultFingerprinter(config.protection.fingerprint)
    fingerprint = fingerprinter.fingerprint_file(args.image, "probe")
    matches: list[dict[str, Any]] = []
    for asset in build_asset_repository(config).list_assets():
        if len(asset.fingerprint.phash) != len(fingerprint.phash):
            continue
        matches.append(
            {
                "asset_id": asset.asset_id,
                "user_id": asset.user_id,
                "exact": asset.fingerprint.sha256 == fingerprint.sha256,
                "phash_similarity": round(
                    hash_similarity(asset.fingerprint.phash, fingerprint.phash), 6
                ),
            }
        )
    matches.sort(key=lambda match: float(match["phash_similarity"]), reverse=True)

    payload = {"fingerprint": fingerprint.to_dict(), "matches": matches[:5]}
    lines = [
        f"fingerprints for {args.image}",
        f"  sha256  {fingerprint.sha256}",
        f"  phash   {fingerprint.phash}",
        f"  dhash   {fingerprint.dhash}",
        "",
        "closest registered assets:" if matches else "no registered assets to compare against",
        *(
            f"  {m['asset_id']}  phash={m['phash_similarity']:.4f}  exact={m['exact']}"
            for m in matches[:5]
        ),
    ]
    _emit(payload, as_json, lines)
    return EXIT_OK


def command_provenance(config: DeepShieldConfig, args: Any, as_json: bool) -> int:
    """Show the recorded lineage of one asset."""
    from deepshield.storage import build_provenance_store

    chain = build_provenance_store(config).lineage(args.asset_id)
    payload = {"asset_id": args.asset_id, "lineage": [r.to_dict() for r in chain]}
    lines = (
        [
            f"lineage for {args.asset_id} (newest first):",
            *(
                f"  {r.asset_id}  sha256={r.sha256[:16]}...  "
                f"watermark={r.watermark_id}  parent={r.parent_asset}"
                for r in chain
            ),
            "",
            "This log records what this system did to a file. It is self-asserted and",
            "says nothing about what happened to the file elsewhere.",
        ]
        if chain
        else [f"no provenance recorded for asset {args.asset_id}"]
    )
    _emit(payload, as_json, lines)
    return EXIT_OK


def command_report(config: DeepShieldConfig, args: Any, as_json: bool) -> int:
    """Render a stored evidence record."""
    from deepshield.storage import build_evidence_repository

    payload = build_evidence_repository(config).get(args.analysis_id)
    if payload is None:
        print(f"error: no stored analysis with id '{args.analysis_id}'", file=sys.stderr)
        return EXIT_ERROR
    _emit(payload, as_json, _render_evidence(payload))
    return EXIT_OK


def command_robustness(config: DeepShieldConfig, args: Any, as_json: bool) -> int:
    """Run one of the transformation robustness benchmarks."""
    from deepshield.experiments import (
        CombinedProtectionExperiment,
        FaceRobustnessExperiment,
        WatermarkRobustnessExperiment,
        load_transformations,
    )
    from deepshield.media import IMAGE_SUFFIXES

    paths: list[Path] = []
    for entry in args.images:
        path = Path(entry)
        if path.is_dir():
            paths.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            )
        else:
            paths.append(path)
    if not paths:
        print("error: no images found", file=sys.stderr)
        return EXIT_ERROR

    pipeline = load_transformations(
        Path("configs/experiments.yaml"), seed=config.runtime.random_seed
    )
    if args.experiment == "face":
        result = FaceRobustnessExperiment(config).run(paths, pipeline)
        value_column = "face_similarity"
    elif args.experiment == "watermark":
        result = WatermarkRobustnessExperiment(config).run(paths, pipeline)
        value_column = "bit_accuracy"
    else:
        result = CombinedProtectionExperiment(config).run(paths)
        value_column = "ssim"

    output = Path(args.output) if args.output else (
        Path(config.experiments.output_dir) / f"{result.experiment_id}.csv"
    )
    result.write_csv(output)

    payload = {
        "experiment_id": result.experiment_id,
        "rows": len(result.rows),
        "output": str(output),
        "summary": result.summary(value_column),
        "environment": result.environment,
    }
    summary: dict[str, float] = result.summary(value_column)
    lines = [
        f"{result.experiment_id}: {len(result.rows)} rows over {len(paths)} images",
        f"  wrote {output}",
        "",
        f"{value_column}: " + (
            f"mean={summary['mean']:.4f} min={summary['min']:.4f} max={summary['max']:.4f}"
            if summary
            else "no numeric values"
        ),
    ]
    _emit(payload, as_json, lines)
    return EXIT_OK


def command_serve(config: DeepShieldConfig, args: Any) -> int:
    """Run the REST API with uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ModelNotAvailableError(
            "uvicorn is not installed; install the 'api' extra: pip install -e '.[api]'"
        ) from exc

    from deepshield.api.app import create_app

    host = args.host or config.api.host
    port = args.port or config.api.port
    print(f"serving DeepShield API on http://{host}:{port} (docs at /docs)")
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")
    return EXIT_OK


def command_not_implemented(name: str) -> int:
    """Report that a recognised command belongs to a later phase."""
    phase = PHASE_OWNERS.get(name, "a later phase")
    error = NotImplementedInPhaseError(f"'deepshield {name}'", phase)
    print(f"error: {error}", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


def _dispatch(config: DeepShieldConfig, args: Any) -> int:
    """Route a parsed command to its handler."""
    as_json = args.json
    command = args.command
    if command == "info":
        return command_info(config, as_json)
    if command == "doctor":
        return command_doctor(config, as_json)
    if command == "download-models":
        return command_download_models(config, args, as_json)
    if command == "enroll":
        return command_enroll(config, args, as_json)
    if command == "identities":
        return command_identities(config, as_json)
    if command == "protect":
        return command_protect(config, args, as_json)
    if command == "analyze-image":
        return command_analyze(config, args, as_json, video=False)
    if command == "analyze-video":
        return command_analyze(config, args, as_json, video=True)
    if command == "watermark-detect":
        return command_watermark_detect(config, args, as_json)
    if command == "fingerprint":
        return command_fingerprint(config, args, as_json)
    if command == "provenance":
        return command_provenance(config, args, as_json)
    if command == "report":
        return command_report(config, args, as_json)
    if command == "robustness-test":
        return command_robustness(config, args, as_json)
    if command == "serve":
        return command_serve(config, args)
    return command_not_implemented(command)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``deepshield`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    try:
        overrides: dict[str, Any] = {}
        if args.log_level:
            overrides["logging"] = {"level": args.log_level}
        config = load_config(
            config_path=args.config,
            thresholds_path=args.thresholds,
            overrides=overrides or None,
        )
    except DeepShieldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    configure_logging(config.logging, force=True)
    logger.debug("cli command=%s", args.command)

    try:
        return _dispatch(config, args)
    except ModelNotAvailableError as exc:
        logger.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MODEL_UNAVAILABLE
    except NotImplementedInPhaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    except DeepShieldError as exc:
        logger.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
