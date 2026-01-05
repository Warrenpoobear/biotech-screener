# tests/test_stochastic_determinism.py
from __future__ import annotations

from decimal import Decimal

from common.random_state import DeterministicRNG, derive_base_seed
from extensions.bootstrap_scoring import compute_bootstrap_ci_decimal


def test_rng_determinism() -> None:
    print("=" * 70)
    print("TEST 1: RNG Determinism")
    print("=" * 70)
    
    base = derive_base_seed("2024-12-15", "test_run", "abc123")
    print(f"Base seed: {base}")
    
    r1 = DeterministicRNG(base, "ctx")
    r2 = DeterministicRNG(base, "ctx")
    
    print(f"Component seed (int): {r1.seed_int}")

    draws1 = [r1.random() for _ in range(1000)]
    draws2 = [r2.random() for _ in range(1000)]

    assert draws1 == draws2, "RNG sequences don't match!"
    assert r1.audit().draws == r2.audit().draws == 1000, "Draw counts don't match!"
    
    print(f"✅ PASS: 1000 random draws are identical")
    print(f"   Audit: {r1.audit().draws} draws from seed {r1.audit().seed_int}")
    print()


def test_bootstrap_determinism() -> None:
    print("=" * 70)
    print("TEST 2: Bootstrap CI Determinism")
    print("=" * 70)
    
    scores = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40"), Decimal("50")]
    base = derive_base_seed("2024-12-15", "test_run", "abc123")
    
    print(f"Input scores: {[str(s) for s in scores]}")
    print(f"Bootstrap samples: 2000")

    rng1 = DeterministicRNG(base, "bootstrap_test")
    out1 = compute_bootstrap_ci_decimal(scores, rng1, n_bootstrap=2000, confidence=Decimal("0.95"))

    rng2 = DeterministicRNG(base, "bootstrap_test")
    out2 = compute_bootstrap_ci_decimal(scores, rng2, n_bootstrap=2000, confidence=Decimal("0.95"))

    assert out1 == out2, "Bootstrap results don't match!"
    
    print(f"✅ PASS: Bootstrap CIs are identical")
    print(f"   Mean: {out1['mean']}")
    print(f"   95% CI: [{out1['ci_lower']}, {out1['ci_upper']}]")
    print(f"   RNG draws: {out1['rng_audit']['draws']}")
    print()


def test_seed_isolation() -> None:
    print("=" * 70)
    print("TEST 3: Seed Isolation (Different Contexts)")
    print("=" * 70)
    
    base = derive_base_seed("2024-12-15", "test_run", "abc123")
    
    # Same base seed, different contexts -> different sequences
    rng_ctx1 = DeterministicRNG(base, "context_A")
    rng_ctx2 = DeterministicRNG(base, "context_B")
    
    draws_ctx1 = [rng_ctx1.random() for _ in range(100)]
    draws_ctx2 = [rng_ctx2.random() for _ in range(100)]
    
    assert draws_ctx1 != draws_ctx2, "Different contexts should produce different sequences!"
    
    print(f"✅ PASS: Different contexts produce different sequences")
    print(f"   Context A seed: {rng_ctx1.seed_int}")
    print(f"   Context B seed: {rng_ctx2.seed_int}")
    print(f"   Sequences differ: {draws_ctx1[0]:.6f} != {draws_ctx2[0]:.6f}")
    print()


def test_date_sensitivity() -> None:
    print("=" * 70)
    print("TEST 4: Date Sensitivity")
    print("=" * 70)
    
    # Different dates -> different base seeds -> different sequences
    base1 = derive_base_seed("2024-12-15", "test_run", "abc123")
    base2 = derive_base_seed("2024-12-16", "test_run", "abc123")
    
    assert base1 != base2, "Different dates should produce different base seeds!"
    
    rng1 = DeterministicRNG(base1, "ctx")
    rng2 = DeterministicRNG(base2, "ctx")
    
    draws1 = [rng1.random() for _ in range(100)]
    draws2 = [rng2.random() for _ in range(100)]
    
    assert draws1 != draws2, "Different dates should produce different sequences!"
    
    print(f"✅ PASS: Different dates produce different sequences")
    print(f"   Date 2024-12-15 base seed: {base1[:16]}...")
    print(f"   Date 2024-12-16 base seed: {base2[:16]}...")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "STOCHASTIC DETERMINISM TEST SUITE" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        test_rng_determinism()
        test_bootstrap_determinism()
        test_seed_isolation()
        test_date_sensitivity()
        
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("✅ All 4 tests PASSED")
        print("=" * 70)
        print()
        print("🎉 Foundation is solid! Ready for Phase 2 integration.")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        raise