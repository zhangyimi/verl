import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


# The lightweight host test environment may not install orjson.  Tracking only
# needs it when the file backend logs, which these W&B initialization tests do
# not exercise.
sys.modules.setdefault("orjson", types.SimpleNamespace(OPT_SERIALIZE_NUMPY=0, dumps=lambda value, option=0: b"{}"))

_TRACKING_PATH = Path(__file__).resolve().parents[2] / "verl" / "utils" / "tracking.py"
_SPEC = importlib.util.spec_from_file_location("_tracking_under_test", _TRACKING_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Tracking = _MODULE.Tracking


class _FakeWandb:
    def __init__(self):
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)

    def finish(self, exit_code=0):
        return None

    @staticmethod
    def Settings(**kwargs):
        return kwargs


class TestTrackingWandbIdentity(unittest.TestCase):
    def test_explicit_identity_is_forwarded(self):
        fake_wandb = _FakeWandb()
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            tracker = Tracking(
                project_name="project",
                experiment_name="display-name",
                default_backend="wandb",
                config={"trainer": {}},
                wandb_run_id="stable-run-id",
                wandb_resume="allow",
            )

        self.assertEqual(fake_wandb.init_calls[0]["id"], "stable-run-id")
        self.assertEqual(fake_wandb.init_calls[0]["resume"], "allow")
        tracker.logger.clear()

    def test_default_does_not_override_wandb_identity(self):
        fake_wandb = _FakeWandb()
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            tracker = Tracking(
                project_name="project",
                experiment_name="display-name",
                default_backend="wandb",
                config={"trainer": {}},
            )

        self.assertNotIn("id", fake_wandb.init_calls[0])
        self.assertNotIn("resume", fake_wandb.init_calls[0])
        tracker.logger.clear()

    def test_must_resume_policy_is_forwarded(self):
        fake_wandb = _FakeWandb()
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            tracker = Tracking(
                project_name="project",
                experiment_name="display-name",
                default_backend="wandb",
                config={"trainer": {}},
                wandb_run_id="existing-run-id",
                wandb_resume="must",
            )

        self.assertEqual(fake_wandb.init_calls[0]["id"], "existing-run-id")
        self.assertEqual(fake_wandb.init_calls[0]["resume"], "must")
        tracker.logger.clear()


if __name__ == "__main__":
    unittest.main()
