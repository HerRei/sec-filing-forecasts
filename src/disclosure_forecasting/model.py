"""Matched generic-sequence and revision-slot models with identical parameters.

Both variants embed every record with the same encoder and pass it through the same GRU
cell. The sequence reference starts each record from a zero state, so a correction is just
another token that attention must relate to the record it replaces. The revision model
updates the state of the record's (metric, economic period) slot in availability order,
so one token per slot carries the version history. Token counts, horizons and the
Transformer are identical; only the fact-memory wiring differs.

fact_budget (Stage S2) caps how many fact tokens reach the Transformer. The sequence model
keeps the newest records; the revision model reads every record into its slots and passes on
the most recently updated slots. Same Transformer cost; with the full budget nothing changes.
"""

import torch
from torch import nn

from .simulator import FACT_FEATURES, HORIZONS, MARKET_FEATURES, MAX_VERSIONS, QUANTILES


class ForecastModel(nn.Module):
    def __init__(self, width=256, layers=6, heads=8, memory="sequence", fact_budget=None):
        super().__init__()
        if memory not in ("sequence", "revision"):
            raise ValueError("memory must be sequence or revision")
        self.memory = memory
        self.fact_budget = fact_budget
        self.fact_encoder = nn.Linear(FACT_FEATURES, width)
        self.market_encoder = nn.Linear(MARKET_FEATURES, width)
        self.update = nn.GRUCell(width, width)
        self.queries = nn.Parameter(torch.randn(len(HORIZONS), width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, len(QUANTILES))
        self.register_buffer("scale", torch.tensor(HORIZONS, dtype=torch.float32).sqrt())

    def fact_memory(self, embeddings, slots, ranks):
        batch, length, width = embeddings.shape
        valid = slots >= 0
        if self.memory == "sequence":
            flat = embeddings.reshape(-1, width)
            states = self.update(flat, torch.zeros_like(flat)).reshape(batch, length, width)
            return states, ~valid
        states = embeddings.new_zeros(batch, length, width)
        # Versions of one slot arrive in availability order; rank r is the r-th version
        # seen. Each pass updates every slot that has an r-th version, all in parallel.
        for rank in range(MAX_VERSIONS):
            rows, columns = torch.nonzero(valid & (ranks == rank), as_tuple=True)
            if rows.numel() == 0:
                continue
            targets = slots[rows, columns]
            new = self.update(embeddings[rows, columns], states[rows, targets])
            states = states.index_put((rows, targets), new.to(states.dtype))
        used = (slots.max(dim=1, keepdim=True).values + 1).clamp_min(0)
        padding = torch.arange(length, device=slots.device)[None, :] >= used
        return states, padding

    def budget(self, states, padding, slots):
        batch, length, width = states.shape
        keep_n = self.fact_budget
        if keep_n is None or keep_n >= length:
            return states, padding
        steps = torch.arange(keep_n, device=states.device)
        if self.memory == "sequence":
            # Records are left-aligned in availability order: keep the newest keep_n.
            count = (~padding).sum(1)
            index = ((count - keep_n).clamp_min(0)[:, None] + steps[None]).clamp(max=length - 1)
            keep = steps[None] < count.clamp(max=keep_n)[:, None]
        else:
            positions = torch.arange(length, device=states.device).expand(batch, length)
            last = torch.full_like(slots, -1).scatter_reduce(
                1, slots.clamp_min(0), torch.where(slots >= 0, positions, -1), "amax"
            )
            index = last.argsort(dim=1, descending=True, stable=True)[:, :keep_n]
            keep = last.gather(1, index) >= 0
        tokens = states.gather(1, index[..., None].expand(-1, -1, width))
        return tokens, ~keep

    def forward(self, facts, slots, ranks, market):
        memory, fact_padding = self.fact_memory(self.fact_encoder(facts), slots, ranks)
        memory, fact_padding = self.budget(memory, fact_padding, slots)
        batch = facts.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat((memory, self.market_encoder(market), queries.to(memory.dtype)), dim=1)
        padding = torch.cat(
            (
                fact_padding,
                torch.zeros(
                    batch, market.shape[1] + len(HORIZONS), dtype=torch.bool, device=facts.device
                ),
            ),
            dim=1,
        )
        hidden = self.encoder(tokens, src_key_padding_mask=padding)
        raw = self.output(self.norm(hidden[:, -len(HORIZONS) :])).float()
        # Monotone quantiles in per-sqrt(session) units, then scaled to each horizon.
        middle = len(QUANTILES) // 2
        steps = nn.functional.softplus(raw)
        lower = steps[..., :middle].flip(-1).cumsum(-1).flip(-1)
        upper = steps[..., middle + 1 :].cumsum(-1)
        median = raw[..., middle : middle + 1]
        quantiles = torch.cat((median - lower, median, median + upper), dim=-1)
        return quantiles * self.scale[None, :, None]


def pinball(prediction, target):
    """Per-example, per-horizon mean pinball over QUANTILES, divided by sqrt(horizon).
    prediction (B, H, Q) and target (B, H) in 1% return units; returns (B, H)."""
    levels = prediction.new_tensor(QUANTILES)
    residual = target.unsqueeze(-1) - prediction
    loss = torch.maximum(levels * residual, (levels - 1) * residual).mean(-1)
    return loss / prediction.new_tensor(HORIZONS).sqrt()


def model_inputs(batch):
    return {key: batch[key] for key in ("facts", "slots", "ranks", "market")}
