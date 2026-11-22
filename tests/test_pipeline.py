"""Fast unit tests (CPU, < 1 min): python -m pytest -q"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ml"), str(ROOT / "backend")]

import eurs_ml  # noqa: E402,F401
from eurs_core import NUM_CLASSES, ModelRegistry  # noqa: E402
from eurs_core.lite import LiteModel, NumpyMLP  # noqa: E402
from eurs_core.onnx_model import OnnxModel  # noqa: E402
from eurs_ml.discover import discover, group_key, split_samples  # noqa: E402
from eurs_ml.io import MaskDecoder  # noqa: E402
from eurs_ml.metrics import confusion, psnr, seg_scores, ssim  # noqa: E402
from eurs_ml.synthetic import generate  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("syn"), n=36, h=60, w=80, seed=1)


def test_metrics_perfect_and_known():
    gt = np.array([[0, 1], [2, 255]], np.uint8)
    s = seg_scores(confusion(gt.copy(), gt))
    assert s["mIoU"] == 1.0 and s["pixel_acc"] == 1.0
    pred = np.array([[0, 0], [2, 3]], np.uint8)  # one coral pixel predicted as background
    s = seg_scores(confusion(pred, gt))
    assert s["per_class"]["Background"]["IoU"] == pytest.approx(0.5)
    assert s["per_class"]["Coral"]["IoU"] == 0.0
    assert s["per_class"]["Elasmobranch"]["IoU"] is None  # absent class excluded from the mean
    assert s["mIoU"] == pytest.approx((0.5 + 0 + 1) / 3)
    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    assert ssim(img, img) == pytest.approx(1.0) and psnr(img, img) > 90


def test_discover_and_group_split(synth):
    samples = discover(synth)
    assert len(samples) == 36 and all(s["clean"] and s["mask"] for s in samples)
    a, rep = split_samples([dict(s) for s in samples], synth, seed=3, min_groups=2)
    b, _ = split_samples([dict(s) for s in samples], synth, seed=3, min_groups=2)
    assert [s["split"] for s in a] == [s["split"] for s in b]  # deterministic
    assert rep["group_aware"]
    groups = {}
    for s in a:
        groups.setdefault(group_key(s["id"]), set()).add(s["split"])
    assert all(len(v) == 1 for v in groups.values())  # no dive leaks across splits
    assert set(rep["counts"]) == {"train", "val", "test"}


def test_mask_decoder_palette_and_index(tmp_path):
    pal = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    m = np.array([[0, 1, 2], [3, 4, 0]], np.uint8)
    rgb = np.array(pal, np.uint8)[m]
    rgb[0, 0] = [9, 9, 8]  # unknown colour -> ignore
    cv2.imwrite(str(tmp_path / "p.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    d = MaskDecoder("palette", pal)
    out = d(tmp_path / "p.png")
    assert out[0, 0] == 255 and (out.ravel()[1:] == m.ravel()[1:]).all() and d.unknown_pixels == 1
    cv2.imwrite(str(tmp_path / "i.png"), np.array([[0, 4, 7]], np.uint8))
    assert MaskDecoder("auto")(tmp_path / "i.png").tolist() == [[0, 4, 255]]


def _tiny_lite():
    from eurs_core.features import restoration_features, segmentation_features
    probe = np.random.rand(16, 16, 3).astype(np.float32)
    nr, ns = restoration_features(probe).shape[2], segmentation_features(probe).shape[2]
    rng = np.random.default_rng(0)
    res = NumpyMLP([rng.normal(0, .01, (nr, 8)).astype(np.float32), rng.normal(0, .01, (8, 3)).astype(np.float32)],
                   [np.zeros(8, np.float32), np.zeros(3, np.float32)], np.zeros(nr, np.float32),
                   np.ones(nr, np.float32), "identity")
    seg = NumpyMLP([rng.normal(0, .1, (ns, NUM_CLASSES)).astype(np.float32)], [np.zeros(NUM_CLASSES, np.float32)],
                   np.zeros(ns, np.float32), np.ones(ns, np.float32), "softmax")
    return LiteModel(res, seg, 64, 1.0, {"dataset": "synthetic"})


def test_lite_roundtrip_and_registry(tmp_path):
    m = _tiny_lite()
    img = (np.random.rand(50, 70, 3) * 255).astype(np.uint8)
    m.save(tmp_path / "eurs_lite.npz")
    m2 = LiteModel.load(tmp_path / "eurs_lite.npz")
    r1, k1 = m.predict(img)
    r2, k2 = m2.predict(img)
    assert r1.shape == img.shape and k1.shape == img.shape[:2]
    assert (r1 == r2).all() and (k1 == k2).all()
    reg = ModelRegistry(tmp_path)
    assert reg.best() == "eurs_lite" and set(reg.models) == {"classical", "eurs_lite"}


def test_onnx_runner_contract(tmp_path):
    class FakeSession:  # mimics onnxruntime: identity restoration, class = argmax of a fixed map
        def run(self, names, feeds):
            x = feeds["image"]
            assert x.dtype == np.float32 and x.shape == (1, 3, 64, 64) and 0 <= x.min() and x.max() <= 1
            logits = np.zeros((1, NUM_CLASSES, 64, 64), np.float32)
            logits[:, 2, :, 32:] = 1
            return [x, logits]
    (tmp_path / "net.json").write_text('{"name": "eurs_net", "input_size": 64}')
    m = OnnxModel(tmp_path / "net.onnx", session=FakeSession())
    img = (np.random.rand(90, 120, 3) * 255).astype(np.uint8)
    restored, mask = m.predict(img)
    assert restored.shape == img.shape and np.abs(restored.astype(int) - img).max() <= 1
    assert mask.shape == (90, 120) and set(np.unique(mask)) == {0, 2}


def test_service_process():
    from app import service
    reg = ModelRegistry(ROOT / "nonexistent-models-dir")
    ok, buf = cv2.imencode(".png", (np.random.rand(40, 50, 3) * 255).astype(np.uint8))
    out = service.process(reg, buf.tobytes(), "a.png")
    assert out["model"] == "classical" and out["width"] == 50 and out["warning"]
    assert abs(sum(out["class_percentages"].values()) - 100) < 0.1
    with pytest.raises(service.ServiceError):
        service.process(reg, b"not an image", "x")
    with pytest.raises(service.ServiceError):
        service.process(reg, buf.tobytes(), "a.png", model="missing")
