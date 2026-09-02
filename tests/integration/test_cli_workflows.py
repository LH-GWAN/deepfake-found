"""CLI end-to-end workflows over a temporary data directory.

Exercises the commands the way a user runs them, against mock face backends so
no model download is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from deepshield.cli import EXIT_ERROR, EXIT_OK, main
from deepshield.media import save_image

pytestmark = pytest.mark.integration


@pytest.fixture
def workspace(tmp_path: Path, project_root: Path):
    """Write a config pointing every store at a temporary directory."""
    base = yaml.safe_load((project_root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    base["runtime"]["data_dir"] = str(tmp_path / "data")
    base["runtime"]["results_dir"] = str(tmp_path / "data" / "results")
    base["runtime"]["model_dir"] = str(tmp_path / "models")
    base["storage"]["embedding_store_dir"] = str(tmp_path / "data" / "embeddings")
    base["face"]["detector"]["backend"] = "mock"
    base["face"]["aligner"]["backend"] = "mock"
    base["face"]["embedder"]["backend"] = "mock"
    base["detection"]["deepfake"]["backend"] = "mock"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return config_path, tmp_path


def run(config_path: Path, *args: str) -> int:
    return main(["--config", str(config_path), "--json", *args])


def make_images(directory: Path, count: int, seed: int = 5, size: int = 512) -> list[Path]:
    from tests.conftest import synthetic_photo

    directory.mkdir(parents=True, exist_ok=True)
    return [
        save_image(synthetic_photo(seed=seed, size=size), directory / f"r{i}.png")
        for i in range(count)
    ]


def read_json(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out)


def test_full_workflow(workspace, capsys: pytest.CaptureFixture[str]) -> None:
    config_path, root = workspace
    references = make_images(root / "refs", 3)

    assert run(config_path, "enroll", str(root / "refs"), "--user-id", "u1") == EXIT_OK
    assert read_json(capsys)["profile"]["user_id"] == "u1"

    assert run(config_path, "identities") == EXIT_OK
    assert read_json(capsys)["identities"][0]["user_id"] == "u1"

    assert run(
        config_path, "protect", str(references[0]), "--user-id", "u1",
        "--distribution-id", "instagram",
    ) == EXIT_OK
    protect = read_json(capsys)
    protected = Path(protect["protected_path"])
    assert protected.is_file()

    assert run(config_path, "watermark-detect", str(protected)) == EXIT_OK
    detection = read_json(capsys)
    assert detection["detected"] is True
    assert detection["watermark_code"] == protect["watermark"]["code"]
    assert detection["matched_asset"]["distribution_id"] == "instagram"

    assert run(config_path, "fingerprint", str(protected)) == EXIT_OK
    fingerprint = read_json(capsys)
    assert fingerprint["matches"][0]["exact"] is True

    assert run(
        config_path, "analyze-image", str(protected), "--user-id", "u1", "--save"
    ) == EXIT_OK
    analysis = read_json(capsys)
    assert analysis["watermark"]["detected"] is True
    assert analysis["fingerprint"]["matched_asset_id"] == protect["asset_id"]
    assert analysis["provenance"]["confidence"] == 1.0
    assert analysis["risk"]["risk_score"] > 0
    assert analysis["limitations"]

    assert run(config_path, "report", analysis["analysis_id"]) == EXIT_OK
    assert read_json(capsys)["analysis_id"] == analysis["analysis_id"]

    assert run(
        config_path, "analyze-image", str(references[1]), "--user-id", "u1"
    ) == EXIT_OK
    identity_hit = read_json(capsys)
    assert identity_hit["identity"]["matched_user_id"] == "u1"
    assert identity_hit["deepfake"]["score"] is not None

    assert run(config_path, "provenance", protect["asset_id"]) == EXIT_OK
    lineage = read_json(capsys)["lineage"]
    assert [record["asset_id"] for record in lineage] == [
        protect["asset_id"],
        f"{protect['asset_id']}:source",
    ]


def test_enroll_rejects_too_few_images(workspace, capsys) -> None:
    config_path, root = workspace
    make_images(root / "few", 1)
    assert run(config_path, "enroll", str(root / "few"), "--user-id", "u1") == EXIT_ERROR
    assert "quality filtering" in capsys.readouterr().err


def test_min_images_override_warns(workspace, capsys) -> None:
    config_path, root = workspace
    make_images(root / "two", 2)
    assert (
        run(config_path, "enroll", str(root / "two"), "--user-id", "u1", "--min-images", "2")
        == EXIT_OK
    )
    assert "less reliable" in capsys.readouterr().err


def test_analysis_without_identity_makes_no_claim(workspace, capsys) -> None:
    config_path, root = workspace
    image = make_images(root / "img", 1)[0]
    assert run(config_path, "analyze-image", str(image)) == EXIT_OK
    payload = read_json(capsys)
    assert payload["identity"]["matched_user_id"] is None
    assert payload["deepfake"]["score"] is None
    assert any("was not run" in line for line in payload["limitations"])


def test_analyze_for_unknown_user_fails_clearly(workspace, capsys) -> None:
    config_path, root = workspace
    image = make_images(root / "img", 1)[0]
    assert run(config_path, "analyze-image", str(image), "--user-id", "ghost") == EXIT_ERROR
    assert "no enrolled identity" in capsys.readouterr().err


def test_robustness_benchmark_writes_csv(workspace, capsys) -> None:
    config_path, root = workspace
    make_images(root / "bench", 1)
    output = root / "bench.csv"
    assert (
        run(
            config_path, "robustness-test", str(root / "bench"),
            "--experiment", "watermark", "--output", str(output),
        )
        == EXIT_OK
    )
    payload = read_json(capsys)
    assert payload["rows"] > 0
    assert output.is_file()
    assert payload["environment"]["git_commit"]


def test_robustness_rejects_an_empty_selection(workspace, capsys) -> None:
    config_path, root = workspace
    (root / "empty").mkdir()
    assert run(config_path, "robustness-test", str(root / "empty")) == EXIT_ERROR
    assert "no images found" in capsys.readouterr().err
