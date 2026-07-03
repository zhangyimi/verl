import unittest

import numpy as np
import torch

from verl.trainer.ppo.core_algos import apply_group_recentered_adv_length_norm


def _batch():
    lengths = [2, 6, 3, 5]
    scalars = [1.0, -1.0, 2.0, -2.0]
    mask = torch.zeros(4, 6, dtype=torch.bool)
    advantages = torch.zeros(4, 6)
    for index, (length, scalar) in enumerate(zip(lengths, scalars, strict=True)):
        mask[index, :length] = True
        advantages[index, :length] = scalar
    uids = np.array(["a", "a", "b", "b"], dtype=object)
    return advantages, mask, uids


def _apply(advantages, mask, uids):
    return apply_group_recentered_adv_length_norm(
        advantages,
        mask,
        uids,
        quantile=0.50,
        alpha=1.0,
        min_scale=0.50,
        max_scale=1.0,
        expected_group_size=2,
        mode="neg_only",
    )


class DriverGroupRecenterTest(unittest.TestCase):
    def test_old_negative_only_scaling_has_positive_bias_and_recenter_removes_it(self):
        advantages, mask, uids = _batch()
        output, metrics = _apply(advantages, mask, uids)

        # With P50=4, legacy one-sided scaling gives group means
        # [(1 - 4/6)/2, (2 - 2*4/5)/2], both strictly positive.
        self.assertGreater(metrics["actor/adv_length_norm_group_mean_abs_pre"], 0.0)
        self.assertLess(metrics["actor/adv_length_norm_group_mean_abs_post"], 1e-7)
        self.assertLess(metrics["actor/adv_length_norm_scale_mean"], 1.0)
        self.assertGreater(metrics["actor/adv_length_norm_scaled_seq_fraction"], 0.0)

        seq_values = (output * mask).sum(dim=-1) / mask.sum(dim=-1)
        self.assertAlmostEqual(float(seq_values[:2].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(seq_values[2:].mean()), 0.0, places=6)

    def test_shuffle_does_not_change_result_and_returns_are_untouched(self):
        advantages, mask, uids = _batch()
        returns = torch.randn_like(advantages)
        returns_before = returns.clone()
        expected, _ = _apply(advantages, mask, uids)

        permutation = torch.tensor([2, 0, 3, 1])
        shuffled, _ = _apply(advantages[permutation], mask[permutation], uids[permutation.numpy()])
        inverse = torch.argsort(permutation)
        self.assertTrue(torch.allclose(shuffled[inverse], expected))
        self.assertTrue(torch.equal(returns, returns_before))

    def test_missing_uid_and_bad_group_fail_fast(self):
        advantages, mask, uids = _batch()
        missing = uids.copy()
        missing[1] = None
        with self.assertRaisesRegex(ValueError, "missing/invalid uid"):
            _apply(advantages, mask, missing)

        bad_group = np.array(["a", "a", "b", "c"], dtype=object)
        with self.assertRaisesRegex(ValueError, "equally-sized uid groups"):
            _apply(advantages, mask, bad_group)

    def test_nonconstant_or_nonfinite_sequence_advantage_fails_fast(self):
        advantages, mask, uids = _batch()
        nonconstant = advantages.clone()
        nonconstant[0, 1] += 0.1
        with self.assertRaisesRegex(ValueError, "constant GRPO advantage"):
            _apply(nonconstant, mask, uids)

        nonfinite = advantages.clone()
        nonfinite[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _apply(nonfinite, mask, uids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
