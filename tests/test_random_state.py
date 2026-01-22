#!/usr/bin/env python3
"""
Tests for common/random_state.py

Tests deterministic random number generation.
Covers:
- Seed derivation
- RNG reproducibility
- Bounds checking
- Bootstrap sampling
- Numpy integration
"""

import pytest
from common.random_state import (
    DeterministicRNG,
    RNGAudit,
    derive_base_seed,
    derive_component_seed_int,
    to_uint32,
    _sha256_hex,
)


class TestSha256Hex:
    """Tests for _sha256_hex helper."""

    def test_sha256_hex_basic(self):
        """Should produce 64 character hex string."""
        result = _sha256_hex("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_sha256_hex_deterministic(self):
        """Same input should produce same output."""
        result1 = _sha256_hex("test input")
        result2 = _sha256_hex("test input")
        assert result1 == result2

    def test_sha256_hex_different_inputs(self):
        """Different inputs should produce different outputs."""
        result1 = _sha256_hex("input1")
        result2 = _sha256_hex("input2")
        assert result1 != result2

    def test_sha256_hex_empty_string(self):
        """Empty string should hash correctly."""
        result = _sha256_hex("")
        assert len(result) == 64


class TestDeriveBaseSeed:
    """Tests for derive_base_seed function."""

    def test_derive_base_seed_format(self):
        """Base seed should be 32 character hex string."""
        result = derive_base_seed("2026-01-15", "run123", "datahash")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_derive_base_seed_deterministic(self):
        """Same inputs should produce same seed."""
        seed1 = derive_base_seed("2026-01-15", "run123", "hash1")
        seed2 = derive_base_seed("2026-01-15", "run123", "hash1")
        assert seed1 == seed2

    def test_derive_base_seed_different_dates(self):
        """Different dates should produce different seeds."""
        seed1 = derive_base_seed("2026-01-15", "run123", "hash1")
        seed2 = derive_base_seed("2026-01-16", "run123", "hash1")
        assert seed1 != seed2

    def test_derive_base_seed_different_run_ids(self):
        """Different run IDs should produce different seeds."""
        seed1 = derive_base_seed("2026-01-15", "run123", "hash1")
        seed2 = derive_base_seed("2026-01-15", "run456", "hash1")
        assert seed1 != seed2

    def test_derive_base_seed_different_data_hashes(self):
        """Different data hashes should produce different seeds."""
        seed1 = derive_base_seed("2026-01-15", "run123", "hash1")
        seed2 = derive_base_seed("2026-01-15", "run123", "hash2")
        assert seed1 != seed2

    def test_derive_base_seed_algo_version(self):
        """Different algo versions should produce different seeds."""
        seed1 = derive_base_seed("2026-01-15", "run123", "hash1", "v1")
        seed2 = derive_base_seed("2026-01-15", "run123", "hash1", "v2")
        assert seed1 != seed2


class TestDeriveComponentSeedInt:
    """Tests for derive_component_seed_int function."""

    def test_derive_component_seed_int_type(self):
        """Should return an integer."""
        result = derive_component_seed_int("baseseed123", "context")
        assert isinstance(result, int)

    def test_derive_component_seed_int_deterministic(self):
        """Same inputs should produce same integer."""
        seed1 = derive_component_seed_int("base", "ctx")
        seed2 = derive_component_seed_int("base", "ctx")
        assert seed1 == seed2

    def test_derive_component_seed_int_different_contexts(self):
        """Different contexts should produce different integers."""
        seed1 = derive_component_seed_int("base", "context1")
        seed2 = derive_component_seed_int("base", "context2")
        assert seed1 != seed2

    def test_derive_component_seed_int_positive(self):
        """Should produce positive integer (16 hex digits)."""
        result = derive_component_seed_int("base", "ctx")
        assert result >= 0


class TestToUint32:
    """Tests for to_uint32 function."""

    def test_to_uint32_small_number(self):
        """Small number should pass through."""
        assert to_uint32(100) == 100

    def test_to_uint32_max_value(self):
        """Max uint32 value should pass through."""
        assert to_uint32(0xFFFFFFFF) == 0xFFFFFFFF

    def test_to_uint32_overflow(self):
        """Larger number should be masked to 32 bits."""
        large_num = 0x1FFFFFFFF  # 33 bits
        result = to_uint32(large_num)
        assert result == 0xFFFFFFFF

    def test_to_uint32_zero(self):
        """Zero should remain zero."""
        assert to_uint32(0) == 0


class TestRNGAudit:
    """Tests for RNGAudit dataclass."""

    def test_rng_audit_creation(self):
        """Should create audit record correctly."""
        audit = RNGAudit(context="test", seed_int=12345, draws=10)
        assert audit.context == "test"
        assert audit.seed_int == 12345
        assert audit.draws == 10

    def test_rng_audit_as_dict(self):
        """as_dict should return proper dictionary."""
        audit = RNGAudit(context="test", seed_int=12345, draws=10)
        d = audit.as_dict()
        assert d == {"context": "test", "seed_int": 12345, "draws": 10}

    def test_rng_audit_immutable(self):
        """RNGAudit is frozen dataclass."""
        audit = RNGAudit(context="test", seed_int=12345, draws=10)
        with pytest.raises(Exception):  # FrozenInstanceError
            audit.draws = 20


class TestDeterministicRNG:
    """Tests for DeterministicRNG class."""

    @pytest.fixture
    def rng(self):
        """Create a standard RNG for testing."""
        return DeterministicRNG(base_seed="testseed123", context="test")

    def test_init_creates_valid_rng(self, rng):
        """Initialization should set up valid RNG."""
        assert rng.context == "test"
        assert isinstance(rng.seed_int, int)
        assert rng._draws == 0

    def test_random_returns_float(self, rng):
        """random() should return float between 0 and 1."""
        result = rng.random()
        assert isinstance(result, float)
        assert 0 <= result < 1

    def test_random_increments_draws(self, rng):
        """random() should increment draw counter."""
        assert rng._draws == 0
        rng.random()
        assert rng._draws == 1
        rng.random()
        assert rng._draws == 2

    def test_random_deterministic(self):
        """Same seed should produce same sequence."""
        rng1 = DeterministicRNG("seed", "ctx")
        rng2 = DeterministicRNG("seed", "ctx")

        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 == seq2

    def test_randint_returns_int(self, rng):
        """randint should return integer in range."""
        result = rng.randint(0, 10)
        assert isinstance(result, int)
        assert 0 <= result < 10

    def test_randint_bounds(self, rng):
        """randint should respect bounds."""
        results = [rng.randint(5, 15) for _ in range(100)]
        assert all(5 <= r < 15 for r in results)
        assert min(results) >= 5
        assert max(results) < 15

    def test_randint_invalid_bounds_raises(self, rng):
        """randint should raise for invalid bounds."""
        with pytest.raises(ValueError, match="high_exclusive must be > low_inclusive"):
            rng.randint(10, 10)  # Equal bounds

        with pytest.raises(ValueError, match="high_exclusive must be > low_inclusive"):
            rng.randint(10, 5)  # Reversed bounds

    def test_choice_returns_element(self, rng):
        """choice should return element from sequence."""
        seq = ["a", "b", "c"]
        result = rng.choice(seq)
        assert result in seq

    def test_choice_empty_sequence_raises(self, rng):
        """choice on empty sequence should raise."""
        with pytest.raises(ValueError, match="empty sequence"):
            rng.choice([])

    def test_choice_deterministic(self):
        """choice should be deterministic."""
        seq = ["a", "b", "c", "d", "e"]

        rng1 = DeterministicRNG("seed", "ctx")
        rng2 = DeterministicRNG("seed", "ctx")

        choices1 = [rng1.choice(seq) for _ in range(20)]
        choices2 = [rng2.choice(seq) for _ in range(20)]

        assert choices1 == choices2

    def test_shuffle_inplace_modifies_list(self, rng):
        """shuffle_inplace should modify the list."""
        original = [1, 2, 3, 4, 5]
        to_shuffle = original.copy()
        rng.shuffle_inplace(to_shuffle)

        # Should contain same elements
        assert sorted(to_shuffle) == sorted(original)
        # Should likely be different order (very unlikely to match)
        # But this is probabilistic, so we just check it runs

    def test_shuffle_inplace_deterministic(self):
        """shuffle_inplace should be deterministic."""
        rng1 = DeterministicRNG("seed", "ctx")
        rng2 = DeterministicRNG("seed", "ctx")

        list1 = [1, 2, 3, 4, 5]
        list2 = [1, 2, 3, 4, 5]

        rng1.shuffle_inplace(list1)
        rng2.shuffle_inplace(list2)

        assert list1 == list2

    def test_bootstrap_sample_indices_length(self, rng):
        """bootstrap_sample_indices should return k indices."""
        result = rng.bootstrap_sample_indices(100, k=50)
        assert len(result) == 50

    def test_bootstrap_sample_indices_default_k(self, rng):
        """bootstrap_sample_indices default k should equal n."""
        result = rng.bootstrap_sample_indices(20)
        assert len(result) == 20

    def test_bootstrap_sample_indices_bounds(self, rng):
        """bootstrap_sample_indices should return indices in [0, n)."""
        n = 50
        result = rng.bootstrap_sample_indices(n, k=100)
        assert all(0 <= i < n for i in result)

    def test_bootstrap_sample_indices_n_zero_raises(self, rng):
        """bootstrap_sample_indices with n=0 should raise."""
        with pytest.raises(ValueError, match="n must be > 0"):
            rng.bootstrap_sample_indices(0)

    def test_bootstrap_sample_indices_negative_n_raises(self, rng):
        """bootstrap_sample_indices with negative n should raise."""
        with pytest.raises(ValueError, match="n must be > 0"):
            rng.bootstrap_sample_indices(-5)

    def test_bootstrap_sample_indices_negative_k_raises(self, rng):
        """bootstrap_sample_indices with negative k should raise."""
        with pytest.raises(ValueError, match="k must be >= 0"):
            rng.bootstrap_sample_indices(10, k=-1)

    def test_bootstrap_sample_indices_k_zero(self, rng):
        """bootstrap_sample_indices with k=0 should return empty list."""
        result = rng.bootstrap_sample_indices(10, k=0)
        assert result == []

    def test_bootstrap_sample_indices_deterministic(self):
        """bootstrap_sample_indices should be deterministic."""
        rng1 = DeterministicRNG("seed", "ctx")
        rng2 = DeterministicRNG("seed", "ctx")

        result1 = rng1.bootstrap_sample_indices(50, k=100)
        result2 = rng2.bootstrap_sample_indices(50, k=100)

        assert result1 == result2

    def test_audit_returns_rng_audit(self, rng):
        """audit() should return RNGAudit with current state."""
        rng.random()
        rng.random()
        rng.random()

        audit = rng.audit()
        assert isinstance(audit, RNGAudit)
        assert audit.context == "test"
        assert audit.draws == 3

    def test_audit_reflects_draws(self, rng):
        """audit should reflect number of draws."""
        assert rng.audit().draws == 0
        rng.random()
        assert rng.audit().draws == 1
        rng.randint(0, 10)
        assert rng.audit().draws == 2
        rng.choice([1, 2, 3])
        assert rng.audit().draws == 3


class TestNumpyIntegration:
    """Tests for numpy RNG integration."""

    def test_numpy_returns_generator_or_none(self):
        """numpy() should return generator or None."""
        rng = DeterministicRNG("seed", "ctx")
        np_rng = rng.numpy()
        # Either numpy is available and returns Generator, or None
        assert np_rng is None or hasattr(np_rng, 'random')

    @pytest.mark.skipif(
        not DeterministicRNG("s", "c")._np,
        reason="NumPy not available"
    )
    def test_numpy_deterministic(self):
        """numpy RNG should be deterministic when available."""
        rng1 = DeterministicRNG("seed", "ctx")
        rng2 = DeterministicRNG("seed", "ctx")

        np1 = rng1.numpy()
        np2 = rng2.numpy()

        if np1 is not None and np2 is not None:
            result1 = [np1.random() for _ in range(10)]
            result2 = [np2.random() for _ in range(10)]
            assert result1 == result2


class TestCrossContextIsolation:
    """Tests for isolation between different contexts."""

    def test_different_contexts_produce_different_sequences(self):
        """Different contexts should produce different sequences."""
        rng1 = DeterministicRNG("same_seed", "context_a")
        rng2 = DeterministicRNG("same_seed", "context_b")

        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 != seq2

    def test_same_context_same_sequence(self):
        """Same context should produce same sequence."""
        rng1 = DeterministicRNG("same_seed", "same_context")
        rng2 = DeterministicRNG("same_seed", "same_context")

        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 == seq2


class TestReproducibility:
    """Tests for overall reproducibility."""

    def test_full_workflow_reproducibility(self):
        """Full workflow should be reproducible."""
        base_seed = derive_base_seed("2026-01-15", "run001", "data_hash_abc")

        # First run
        rng1 = DeterministicRNG(base_seed, "scoring")
        result1 = {
            "random": [rng1.random() for _ in range(5)],
            "ints": [rng1.randint(0, 100) for _ in range(5)],
            "bootstrap": rng1.bootstrap_sample_indices(20, k=10),
        }

        # Second run with same seed
        rng2 = DeterministicRNG(base_seed, "scoring")
        result2 = {
            "random": [rng2.random() for _ in range(5)],
            "ints": [rng2.randint(0, 100) for _ in range(5)],
            "bootstrap": rng2.bootstrap_sample_indices(20, k=10),
        }

        assert result1 == result2
