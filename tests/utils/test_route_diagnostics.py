import pytest
import torch
from torch.utils.checkpoint import checkpoint

from verl.utils.route_diagnostics import (
    FSDPRouteCapture,
    compact_routed_experts,
    finalize_route_diag_metrics,
    route_replay_counterfactual_metrics,
)


def test_compact_routed_experts_validates_and_uses_one_byte_ids():
    routes = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.int64)
    compact = compact_routed_experts(routes)
    assert compact.dtype == torch.uint8
    assert torch.equal(compact.long(), routes)

    with pytest.raises(RuntimeError, match="all-zero"):
        compact_routed_experts(torch.zeros_like(routes))


class _Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 16, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Mlp()


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()

    def forward(self, hidden):
        return [layer.mlp.gate(hidden) for layer in self.model.layers]


def test_route_capture_and_finalize_prompt_response_scopes():
    torch.manual_seed(7)
    model = _ToyModel()
    capture = FSDPRouteCapture(model, topk=8)
    capture.begin()
    model(torch.randn(7, 4))

    natural = torch.stack([capture._routes[0], capture._routes[1]], dim=1)
    rollout = natural.clone()
    # Sequence 1 is length 4 with loss mask [prompt,prompt,response,response].
    # Position 2 is a response predictor position; force one expert-set mismatch there.
    replacement = next(i for i in range(16) if i not in rollout[2, 1].tolist())
    rollout[2, 1, 0] = replacement

    rollout_nested = torch.nested.as_nested_tensor([rollout[:4], rollout[4:]], layout=torch.jagged)
    loss_mask = torch.nested.as_nested_tensor(
        [torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1])], layout=torch.jagged
    )
    raw = capture.finish({"routed_experts": rollout_nested, "loss_mask": loss_mask})
    metrics = finalize_route_diag_metrics({key: [[value]] for key, value in raw.items()})

    assert metrics["route_diag/prompt/set_overlap_mean"] == 1.0
    assert metrics["route_diag/prompt/exact_match_mean"] == 1.0
    assert metrics["route_diag/response/set_overlap_mean"] < 1.0
    assert metrics["route_diag/response/exact_match_mean"] < 1.0
    assert metrics["route_diag/num_layers"] == 2.0


def test_route_capture_accepts_dense_response_only_loss_mask():
    torch.manual_seed(11)
    model = _ToyModel()
    capture = FSDPRouteCapture(model, topk=8)
    capture.begin()
    model(torch.randn(7, 4))

    natural = torch.stack([capture._routes[0], capture._routes[1]], dim=1)
    rollout = natural.clone()
    # Sequence lengths are [4, 3], with two response tokens per sequence.
    # The response predictor positions are [1, 2] and [4, 5].
    replacement = next(i for i in range(16) if i not in rollout[2, 1].tolist())
    rollout[2, 1, 0] = replacement
    rollout_nested = torch.nested.as_nested_tensor([rollout[:4], rollout[4:]], layout=torch.jagged)
    dense_response_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])

    raw = capture.finish({"routed_experts": rollout_nested, "loss_mask": dense_response_mask})
    metrics = finalize_route_diag_metrics({key: [[value]] for key, value in raw.items()})

    assert metrics["route_diag/prompt/set_overlap_mean"] == 1.0
    assert metrics["route_diag/response/set_overlap_mean"] < 1.0
    assert metrics["route_diag/response/token_count"] == 4.0


def test_route_capture_can_force_rollout_expert_sets_and_preserve_selected_weights():
    torch.manual_seed(19)
    model = _ToyModel()
    capture = FSDPRouteCapture(model, topk=8)
    hidden = torch.randn(7, 4)
    natural_outputs = model(hidden)

    # Use each gate's bottom eight experts as a deliberately different replay
    # set.  The final all-zero row models the last generated token, for which
    # vLLM has no route and FSDP must retain its natural route.
    forced = torch.stack([torch.topk(scores, k=8, largest=False).indices for scores in natural_outputs], dim=1)
    forced[-1].zero_()
    capture.begin(
        {"routed_experts": forced},
        force_rollout_routes=True,
        collect_metrics=False,
    )
    replay_outputs = model(hidden)
    assert capture.finish({}) == {}

    for layer, (natural_scores, replay_scores) in enumerate(zip(natural_outputs, replay_outputs, strict=True)):
        replay_topk = torch.topk(replay_scores, k=8).indices
        assert torch.equal(replay_topk[:-1].sort(dim=-1).values, forced[:-1, layer].sort(dim=-1).values)
        natural_topk_last = torch.topk(natural_scores[-1], k=8).indices.sort().values
        assert torch.equal(replay_topk[-1].sort().values, natural_topk_last)

        selected_ids = forced[:-1, layer].long()
        expected_weights = torch.softmax(natural_scores[:-1].gather(1, selected_ids).float(), dim=-1)
        replay_weights = torch.softmax(replay_scores[:-1].float(), dim=-1).gather(1, selected_ids)
        assert torch.allclose(replay_weights, expected_weights, atol=1e-6, rtol=1e-6)


def test_route_replay_spans_reentrant_backward_and_clears_between_microbatches():
    torch.manual_seed(23)
    model = _ToyModel()
    capture = FSDPRouteCapture(model, topk=8)
    hidden = torch.randn(5, 4)

    first_routes = torch.arange(8, dtype=torch.uint8).view(1, 1, 8).expand(5, 2, 8).clone()
    capture.begin(
        {"routed_experts": first_routes},
        force_rollout_routes=True,
        collect_metrics=False,
        allow_reentry=True,
        require_checkpoint_reentry=True,
    )
    def checkpointed_forward(value):
        return torch.stack(model(value), dim=0)

    # Non-reentrant checkpointing re-executes the gates during backward.  The
    # replay scope must remain active and reuse the same route set.
    replay_outputs = checkpoint(checkpointed_forward, hidden, use_reentrant=False)
    loss = torch.logsumexp(replay_outputs, dim=-1).sum()
    loss.backward()
    assert all(count >= 2 for count in capture._execution_counts.values())
    capture.end_replay()

    for layer in model.model.layers:
        grad = layer.mlp.gate.weight.grad
        assert grad[:8].abs().sum() > 0
        assert torch.count_nonzero(grad[8:]) == 0

    second_routes = torch.arange(8, 16, dtype=torch.uint8).view(1, 1, 8).expand(5, 2, 8).clone()
    capture.begin(
        {"routed_experts": second_routes},
        force_rollout_routes=True,
        collect_metrics=False,
    )
    second_outputs = model(hidden)
    capture.finish({})
    for layer, scores in enumerate(second_outputs):
        assert torch.equal(
            torch.topk(scores, k=8).indices.sort(dim=-1).values,
            second_routes[:, layer].long().sort(dim=-1).values,
        )


def test_route_replay_fails_if_checkpoint_recomputation_did_not_run():
    model = _ToyModel()
    capture = FSDPRouteCapture(model, topk=8)
    hidden = torch.randn(3, 4)
    routes = torch.arange(8, dtype=torch.uint8).view(1, 1, 8).expand(3, 2, 8).clone()
    capture.begin(
        {"routed_experts": routes},
        force_rollout_routes=True,
        collect_metrics=False,
        allow_reentry=True,
        require_checkpoint_reentry=True,
    )
    model(hidden)
    with pytest.raises(RuntimeError, match="forward and backward recomputation"):
        capture.end_replay()


def test_production_r3_rejects_missing_predictor_routes():
    torch.manual_seed(29)
    model = _ToyModel()
    capture = FSDPRouteCapture(model, topk=8)
    routes = torch.arange(8, dtype=torch.uint8).view(1, 1, 8).expand(4, 2, 8).clone()
    routes[2].zero_()
    routes_nested = torch.nested.as_nested_tensor([routes], layout=torch.jagged)
    response_mask = torch.tensor([[1, 1, 0]])

    with pytest.raises(RuntimeError, match="complete prompt/response-predictor routes"):
        capture.begin(
            {"routed_experts": routes_nested, "loss_mask": response_mask},
            force_rollout_routes=True,
            collect_metrics=False,
            require_complete_routes=True,
        )


def test_route_replay_counterfactual_metrics_report_recovery():
    rollout = torch.zeros(2, 3)
    natural = torch.tensor([[1.0, -2.0, 99.0], [3.0, 99.0, 99.0]])
    replay = torch.tensor([[0.25, -0.5, -99.0], [0.75, -99.0, -99.0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    metrics = route_replay_counterfactual_metrics(rollout, natural, replay, mask)

    assert metrics["route_replay_cf/token_count"] == 3.0
    assert metrics["route_replay_cf/improved_token_fraction"] == 1.0
    assert metrics["route_replay_cf/natural/logprob_abs/mean"] == 2.0
    assert metrics["route_replay_cf/replay/logprob_abs/mean"] == 0.5
    assert metrics["route_replay_cf/logprob_abs_mean_recovery"] == 0.75
