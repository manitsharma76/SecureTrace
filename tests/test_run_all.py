import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_all", Path(__file__).resolve().parents[1] / "scripts" / "run_all.py")
run_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_all)


def test_steps_run_in_dependency_order_and_public_retraining_is_opt_in():
    names = [n for n, _ in run_all.steps(False)]
    assert names == ["gonogo", "labelquality", "before", "adapt", "unsup", "fusion"]
    assert [n for n, _ in run_all.steps(True)][0] == "public"


def test_before_step_covers_every_family_for_both_held_out_mutators():
    cmds = dict(run_all.steps(False))["before"]
    assert len(cmds) == len(run_all.FAMILIES) * len(run_all.MUTATORS)
