import numpy as np
import pytest
import torch

from disclosure_forecasting.model import ForecastModel, model_inputs, pinball
from disclosure_forecasting.simulator import encode, simulate

torch.set_num_threads(1)


def batch(seeds, family="varied", decision=180):
    rows = [encode(simulate(s, family), decision) for s in seeds]
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0]}


def reference_slots(model, embeddings, slots):
    """Plain loop: one GRU update per record, in availability order, per slot."""
    out = torch.zeros_like(embeddings)
    for row in range(embeddings.shape[0]):
        memory = {}
        for index, slot in enumerate(slots[row].tolist()):
            if slot >= 0:
                previous = memory.get(slot, embeddings.new_zeros(embeddings.shape[-1]))
                memory[slot] = model.update(embeddings[row, index][None], previous[None])[0]
        for slot, state in memory.items():
            out[row, slot] = state
    return out


def test_vectorised_revision_memory_matches_the_reference_loop():
    torch.manual_seed(3)
    model = ForecastModel(width=32, layers=1, heads=4, memory="revision").eval()
    data = batch([1, 2, 3, 4], "h_late_revisions")
    assert (data["ranks"] == 2).any()  # third versions are exercised
    with torch.no_grad():
        embeddings = model.fact_encoder(data["facts"])
        states, padding = model.fact_memory(embeddings, data["slots"], data["ranks"])
        expected = reference_slots(model, embeddings, data["slots"])
    torch.testing.assert_close(states, expected)
    slots_used = data["slots"].max(1).values + 1
    assert torch.equal((~padding).sum(1), slots_used)


@pytest.mark.parametrize("memory", ["sequence", "revision"])
def test_quantiles_are_ordered_and_padding_cannot_change_predictions(memory):
    torch.manual_seed(12)
    model = ForecastModel(width=32, layers=1, heads=4, memory=memory).eval()
    data = batch([5, 6])
    with torch.no_grad():
        prediction = model(**model_inputs(data))
        modified = {k: v.clone() for k, v in data.items()}
        modified["facts"][data["slots"] < 0] = 900
        again = model(**model_inputs(modified))
    assert prediction.shape == (2, 3, 7)
    assert torch.all(prediction.diff(dim=-1) > 0)
    torch.testing.assert_close(prediction, again)
    assert pinball(prediction, data["target"]).shape == (2, 3)


def test_variants_have_identical_parameters():
    sequence = ForecastModel(width=32, layers=1, heads=4)
    revision = ForecastModel(width=32, layers=1, heads=4, memory="revision")

    def shapes(m):
        return [(k, v.shape) for k, v in m.state_dict().items()]

    assert shapes(sequence) == shapes(revision)


def test_pinball_is_minimised_by_the_true_quantiles():
    torch.manual_seed(0)
    target = torch.randn(20000, 3)
    levels = torch.tensor((0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95))
    true = torch.distributions.Normal(0, 1).icdf(levels).expand(20000, 3, 7)
    assert pinball(true, target).mean() < pinball(true * 1.3, target).mean()
    assert pinball(true, target).mean() < pinball(true + 0.2, target).mean()
