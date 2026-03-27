from src.ingest.shot_detector import detect_shots


def test_hard_cut_detection():
    diffs = [0.05, 0.07, 0.9, 0.04, 0.03]
    shots = detect_shots(diffs, fps=10.0, duration=0.6)
    assert len(shots) == 2
    assert shots[1].transition_type == "hard_cut"
    assert shots[0].end_time == 0.3
