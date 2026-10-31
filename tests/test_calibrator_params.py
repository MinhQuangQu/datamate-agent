import numpy as np
from safetensors.numpy import save_file

from dma.utils.count_params import count_dir, main as count_main
from dma.verify.calibrator import Calibrator, choose_threshold, expected_calibration_error, pa_curve


def test_threshold_and_fit(tmp_path):
    rng = np.random.default_rng(0)
    X, y = [], []
    for _ in range(4000):
        agree = int(rng.random() < 0.7)
        err = int(rng.integers(0, 3))
        ent = rng.random()
        z = -1.0 + 3.0 * agree - 1.2 * err + 1.5 * ent          # logistic ground truth
        X.append({"agree_ab": agree, "exec_errors": err, "entity_score": ent})
        y.append(int(rng.random() < 1 / (1 + np.exp(-z))))
    cal = Calibrator().fit(X, y)
    probs = [cal.prob(x) for x in X]
    assert cal.prob({"agree_ab": 1, "exec_errors": 0}) > cal.prob({"agree_ab": 0, "exec_errors": 2})
    t, best = choose_threshold(probs, y)
    assert t >= 0.5
    answer_all = pa_curve(probs, y, np.array([0.0]))[0][1]
    assert best >= answer_all                         # abstaining never hurts at the optimum
    assert expected_calibration_error(probs, y) < 0.05
    path = tmp_path / "cal.json"
    cal.save(str(path))
    assert abs(Calibrator.load(str(path)).prob(X[0]) - probs[0]) < 1e-12


def test_count_params(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    save_file({"model.embed.weight": np.zeros((100, 8), np.float16),
               "model.layers.0.w": np.zeros((8, 8), np.float16)}, str(d / "a.safetensors"))
    save_file({"model.visual.proj": np.zeros((4, 8), np.float16)}, str(d / "b.safetensors"))
    r = count_dir(str(d))
    assert r["total"] == 800 + 64 + 32
    assert count_main([str(d), "--limit", "500"]) == 2      # over the limit -> non-zero exit
