# common/random_state.py
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Optional


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def derive_base_seed(as_of_date: str, run_id: str, data_hash: str, algo_version: str = "v1") -> str:
    combined = f"{algo_version}|{as_of_date}|{run_id}|{data_hash}"
    return _sha256_hex(combined)[:32]


def derive_component_seed_int(base_seed: str, context: str) -> int:
    combined = f"{base_seed}:{context}"
    return int(_sha256_hex(combined)[:16], 16)


def to_uint32(seed_int: int) -> int:
    return seed_int & 0xFFFFFFFF


@dataclass(frozen=True)
class RNGAudit:
    context: str
    seed_int: int
    draws: int

    def as_dict(self) -> dict:
        return {"context": self.context, "seed_int": self.seed_int, "draws": self.draws}


class DeterministicRNG:
    def __init__(self, base_seed: str, context: str):
        self.context = context
        self.seed_int = derive_component_seed_int(base_seed, context)
        self._py = random.Random(self.seed_int)
        self._draws = 0

        self._np = None
        try:
            import numpy as np
            self._np = np.random.Generator(np.random.PCG64(to_uint32(self.seed_int)))
        except Exception:
            self._np = None

    def random(self) -> float:
        self._draws += 1
        return self._py.random()

    def randint(self, low_inclusive: int, high_exclusive: int) -> int:
        if high_exclusive <= low_inclusive:
            raise ValueError("high_exclusive must be > low_inclusive")
        self._draws += 1
        return self._py.randrange(low_inclusive, high_exclusive)

    def choice(self, seq: list[Any]) -> Any:
        if not seq:
            raise ValueError("choice() arg is an empty sequence")
        self._draws += 1
        return seq[self._py.randrange(0, len(seq))]

    def shuffle_inplace(self, seq: list[Any]) -> None:
        self._draws += 1
        self._py.shuffle(seq)

    def bootstrap_sample_indices(self, n: int, k: Optional[int] = None) -> list[int]:
        if n <= 0:
            raise ValueError("n must be > 0")
        if k is None:
            k = n
        if k < 0:
            raise ValueError("k must be >= 0")

        out = []
        for _ in range(k):
            out.append(self.randint(0, n))
        return out

    def numpy(self) -> Any:
        return self._np

    def audit(self) -> RNGAudit:
        return RNGAudit(context=self.context, seed_int=self.seed_int, draws=self._draws)