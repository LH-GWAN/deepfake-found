"""Phase 14: REST surface, error mapping and disclosure of limits."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from deepshield.config import default_config

pytestmark = pytest.mark.integration
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from deepshield.api.app import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path: Path):
    base = default_config()
    config = base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={
                    "data_dir": tmp_path / "data",
                    "results_dir": tmp_path / "data" / "results",
                    "model_dir": tmp_path / "models",
                }
            ),
            "storage": base.storage.model_copy(
                update={"embedding_store_dir": tmp_path / "data" / "embeddings"}
            ),
            "face": base.face.model_copy(
                update={
                    "detector": base.face.detector.model_copy(update={"backend": "mock"}),
                    "aligner": base.face.aligner.model_copy(update={"backend": "mock"}),
                    "embedder": base.face.embedder.model_copy(update={"backend": "mock"}),
                }
            ),
            "detection": base.detection.model_copy(
                update={"deepfake": base.detection.deepfake.model_copy(update={"backend": "mock"})}
            ),
            "protection": base.protection.model_copy(
                update={
                    "watermark": base.protection.watermark.model_copy(update={"backend": "dct"})
                }
            ),
        }
    )
    return TestClient(create_app(config))


def png_bytes(seed: int = 5, size: int = 512) -> bytes:
    from tests.conftest import synthetic_photo

    buffer = io.BytesIO()
    Image.fromarray(synthetic_photo(seed=seed, size=size)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_backends_and_calibration(client) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["backends"]["face_embedder"] == "mock"
    assert payload["thresholds_calibrated"]["deepfake"] is False


def test_limitations_endpoint_states_the_attribution_limit(client) -> None:
    payload = client.get("/limitations").json()
    assert "training data" in payload["cannot_prove"]


def test_openapi_description_states_the_limit(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "training data" in schema["info"]["description"]


def test_enroll_then_list(client) -> None:
    files = [("files", (f"r{i}.png", png_bytes(seed=5, size=300), "image/png")) for i in range(3)]
    response = client.post("/identity/enroll", data={"user_id": "u1"}, files=files)
    assert response.status_code == 200
    assert response.json()["profile"]["user_id"] == "u1"
    assert client.get("/identity").json()["identities"][0]["user_id"] == "u1"


def test_enroll_with_too_few_images_is_422(client) -> None:
    files = [("files", ("r0.png", png_bytes(), "image/png"))]
    response = client.post("/identity/enroll", data={"user_id": "u1"}, files=files)
    assert response.status_code == 422
    assert "quality filtering" in response.json()["error"]


def test_protect_then_detect_watermark(client) -> None:
    image = png_bytes()
    protect = client.post(
        "/protect/image",
        data={"user_id": "u1", "distribution_id": "instagram"},
        files={"file": ("a.png", image, "image/png")},
    )
    assert protect.status_code == 200
    code = protect.json()["watermark"]["code"]

    protected_path = Path(protect.json()["protected_path"])
    detect = client.post(
        "/watermark/detect",
        files={"file": ("p.png", protected_path.read_bytes(), "image/png")},
    )
    payload = detect.json()
    assert payload["detected"] is True
    assert payload["watermark_code"] == code
    assert payload["matched_asset"]["distribution_id"] == "instagram"
    assert "inconclusive" in payload["interpretation"]


def test_analyze_image_returns_evidence_with_limitations(client) -> None:
    response = client.post(
        "/analyze/image", files={"file": ("a.png", png_bytes(), "image/png")}
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["analysis_id"]
    assert payload["limitations"]
    assert payload["risk"] is not None


def test_stored_analysis_can_be_fetched(client) -> None:
    analysis_id = client.post(
        "/analyze/image", files={"file": ("a.png", png_bytes(), "image/png")}
    ).json()["analysis_id"]
    assert client.get(f"/analysis/{analysis_id}").status_code == 200


def test_unknown_analysis_is_404(client) -> None:
    assert client.get("/analysis/does-not-exist").status_code == 404


def test_corrupt_upload_is_415(client) -> None:
    response = client.post(
        "/analyze/image", files={"file": ("a.png", b"not an image", "image/png")}
    )
    assert response.status_code == 415


def test_analyze_for_unknown_user_is_404(client) -> None:
    response = client.post(
        "/analyze/image",
        data={"user_id": "nobody"},
        files={"file": ("a.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 404
