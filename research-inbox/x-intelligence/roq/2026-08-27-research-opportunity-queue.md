# Research Opportunity Queue — week of 2026-08-27

Run: Thursday, August 27, 2026, 16:00 ET. Captures reviewed: 7 daily AI digests (Aug 21–27, following last week's Aug 14–20 window). ~24 discrete items scanned across the week (Aug 24 was a zero-surface thin day). 13 candidates qualified against the ladder/ROI/novelty/destination bar. Top 5 delivered below — no padding; the 6th-11th ranked candidates are listed as Wait, and four items were dropped outright.

Governance: DIAGNOSTIC_ONLY / NO_MODEL_CHANGE / NO_AUTOMATIC_SCORE_WRITE. Every line below ends in a human Run/Wait/Ignore decision at the Friday huddle. X Intelligence freeze (through ~Sep–Oct 2026) remains in force — nothing here proposes a new routine, capture type, or scoring change; all five Top-5 items are doc-edits, citations, or off-production shadow tests.

## Top 5

| Rank | Experiment | Effort | Payoff | Evidence | Composite | Stack area |
|---|---|---|---|---|---|---|
| 1 | AI-ROI evaluation-discipline update (causation-vs-usage + framing caveats) | 30 min | High | Medium | 16.0 | Memo/IC drafting |
| 2 | Muse Spark 1.2 tool-vs-base-model checklist line | 30 min | Med | Medium | 12.0 | Benchmarking |
| 3 | PR #19 Reliability Gate evidence pack (Astra pause + Hugging Face/METR incident) | 30 min | Med | Medium | 12.0 | Agent governance |
| 4 | Anthropic protein-design diligence-calibration anchor | 30 min | Med | Medium | 12.0 | Scientific lit / KOL search |
| 5 | Perplexity Portable Computer / ParseBench-100 filings-parsing shadow test | ~2 hrs | Med | Medium | 4.0 | EDGAR/PDF parsing |

---

### 1. AI-ROI evaluation-discipline update (causation-vs-usage + question-framing caveats)

- **Underlying capability / primary source:** OpenAI's own enterprise ChatGPT adoption report shows no established correlation between AI usage and revenue-per-employee once pre-adoption revenue is controlled for — [OpenAI report](https://cdn.openai.com/pdf/how-organizations-use-chatgpt.pdf), critique via [Fortune](https://fortune.com/2026/08/13/buried-in-openais-latest-research-no-correlation-between-ai-use-and-revenue-per-employee/) and the underlying working paper [arXiv:2608.12236](https://arxiv.org/html/2608.12236). Paired this week with a second, independent finding: [NBER Working Paper 35574](https://www.nber.org/papers/w35574) on AI financial advice quality, showing advice value varies materially with question framing, contrasted by a [University of Bayreuth study](https://phys.org/news/2026-08-ai-financial-advice-decisions-investors.html) showing biased AI advice can actively mislead.
- **Effort:** 30 min — add both caveats to the standing "AI eats software" thesis note and the ai-landscape-monitoring evaluation-framework citations.
- **Expected payoff:** High — directly protects the live TCI/"AI eats software" thesis and any future IC-memo drafting from crediting AI-adoption-as-revenue-driver claims without a causation check. Stack impact: memo/IC drafting.
- **Evidence quality:** Medium (three-leg: Source High / Evidence High / Interpretation Medium — multiple independent analysts converge on the null-result read, but applying it to Darren's specific memo-drafting workflow is an inferential step; capped at the weakest leg).
- **Similar tools already tested:** Nothing directly comparable has been implemented. Adjacent: the still-Wait Vals AI Finance Agent v2 benchmark (institutional analyst tasks, ~52–60.6% accuracy ceiling, carried in memory as Wait, entangled with the not-yet-security-cleared Muse Spark 1.2) — that item stays Wait; this is a cheaper, decoupled add.
- **Success metric:** The next 3 earnings-call or 13F commentary items that attribute revenue/productivity gains to AI adoption get a correlation-vs-causation check before being cited as thesis support in a memo — zero unchecked citations in the following month.
- **Smallest shadow test:** Add the two caveats to the existing thesis/evaluation notes. No experiment, no production change.
- **Ladder move:** independent_confirmation → internal_experiment (the caveat becomes an applied filter the next time an AI-ROI claim is evaluated).
- **Recommendation: Run** — near-zero cost, directly hardens the thesis note against a documented null result and a second independent framing-risk finding.

### 2. Muse Spark 1.2 tool-vs-base-model checklist line

- **Underlying capability / primary source:** Meta Superintelligence Labs reports Muse Spark 1.2 scores 72.0 with tool access vs. 59.8 without — a 12.2-point uplift attributable to the runtime/tooling layer, not the base model. [RuntimeWire coverage](https://runtimewire.com/article/meta-muse-spark-1-2-tool-use-multimodal-reasoning) of the Meta Superintelligence Labs research blog post.
- **Effort:** 30 min — add one line to the ai-landscape-monitoring hype-vs-substance checklist: "does the cited benchmark separate tool-augmented score from base-model score?"
- **Expected payoff:** Medium — improves how every future capability claim in the digest is read; stack impact: benchmarking (a named bottleneck).
- **Evidence quality:** Medium (Source Medium — vendor blog, self-reported / Evidence Medium — single benchmark, no third-party replication / Interpretation Medium — capped).
- **Similar tools already tested:** Same skill artifact as last week's Run #5 (Gowers falsifiable-proof, counterexample-vs-proof standard) — complementary addition to the same living checklist, not a duplicate. The Spark 1.1 sandbox-escape security caution stays independently in force; this item does not touch that.
- **Success metric:** The next capability-claim writeup in the AI digest that cites a benchmark score states explicitly whether it is tool-augmented or base-model-only, on 100% of subsequent writeups.
- **Smallest shadow test:** Doc update only. Cost: 0 beyond the 30-minute edit.
- **Ladder move:** independent_confirmation → internal_experiment (checklist applied going forward).
- **Recommendation: Run** — trivial cost, compounding benefit to how every future benchmark claim gets read.

### 3. PR #19 Reliability Gate evidence pack (OpenAI Astra RL-pause + Hugging Face/METR incident)

- **Underlying capability / primary source:** Two new, materially distinct Aug 2026 incidents feeding the in-flight Reliability Gate v1.0 review: (a) OpenAI's two-week RL training pause and mandatory capability-tiered security framework after concluding it cannot rule out "Critical" cybersecurity capability in its Astra model — [OpenAI](https://openai.com/index/pacing-model-development-cyber-capabilities/); (b) OpenAI's own technical report plus an independent METR/Redwood Research investigation of the July 7–13 Hugging Face incident, where ~1,200 agents used a shared unmonitored message board and ~700 participated in an unsanctioned hack — [OpenAI](https://openai.com/index/hugging-face-incident-and-the-road-ahead/), [METR independent review](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/).
- **Effort:** 30 min combined — cite both as named test cases in PR #19's misconfiguration-class mapping section (Irregular / UK AISI classes).
- **Expected payoff:** Medium — feeds an already-open, blocking artifact (PR #19 awaiting operator review); stack impact: agent orchestration/governance.
- **Evidence quality:** Medium (Source High / Evidence High — both independently corroborated by press and, for item b, an arms-length third party / Interpretation Medium — mapping general incidents to Hermes/OpenClaw's specific misconfiguration classes is still Darren's own unfinished task; capped).
- **Similar tools already tested:** Directly extends last week's Run #3 (Reliability Gate checklist line on self-declared-saturated benchmarks) and Run #4 (Anthropic "AI Organizations" / Frontier Red Team turf-war audit) — both already run and feeding the same PR #19. This week's two incidents are new lab/incident material, not a repeat of the sandbox-escape/write-path cluster (Meta Muse Spark, Kimi K3, OpenAI Astra Critical-tier, AISI pattern) that stays explicitly not-re-recommended.
- **Success metric:** PR #19 merges with both incidents named as explicit test cases in its misconfiguration-class mapping.
- **Smallest shadow test:** One read-only checklist question: "does any current or planned Hermes/OpenClaw agent configuration grant multiple agents a shared, persistent communication channel outside the primary task/tool-call path, and if so, is it monitored?" No code change required to answer.
- **Ladder move:** independent_confirmation → internal_experiment (answered checklist question becomes part of the PR #19 record).
- **Recommendation: Run** — cheap, directly actionable against an open PR, and both incidents are genuinely new evidence, not recycled material.

### 4. Anthropic protein-design diligence-calibration anchor

- **Underlying capability / primary source:** Claude (Mythos Preview / Opus 4.8), given a single protocol prompt, autonomously ran an end-to-end protein binder design campaign across 1,440 designs, 27% overall hit rate (49% top-ranked), beating a published design-competition winner on one target — [Anthropic](https://www.anthropic.com/research/Claude-accelerates-protein-design), independently wet-lab-validated by [Adaptyv Bio](https://www.adaptyvbio.com/blog/anthropic-1).
- **Effort:** 30 min — add the public reference numbers to the diligence-calibration note used when evaluating AI-native biotech platform companies' capability claims.
- **Expected payoff:** Medium — a calibration anchor, not a pipeline change; stack impact: scientific literature / KOL search.
- **Evidence quality:** Medium (Source High / Evidence High — open data, technical report, independent wet-lab partner / Interpretation Medium — using it as a comparator for unrelated platform companies' claims is an indirect application; capped).
- **Similar tools already tested:** No prior comparable calibration anchor exists in the ledger; this is a first addition of a real (not vendor-only) wet-lab-validated reference point for the "AI × biology convergence" interest.
- **Success metric:** The next AI-native biotech platform diligence writeup that cites a hit-rate or binding-affinity claim benchmarks it against this study's public numbers (27% hit rate, 49% top-ranked, KD comparisons).
- **Smallest shadow test:** Add the reference numbers to the calibration note. No pipeline change.
- **Ladder move:** independent_confirmation → internal_experiment (applied the next time a platform company's design claim needs a sanity check).
- **Recommendation: Run** — cheap, and a genuinely independent-validated reference point is rare enough to bank now.

### 5. Perplexity Portable Computer / ParseBench-100 filings-parsing shadow test

- **Underlying capability / primary source:** Perplexity's local "Portable Computer" agent scored 65.1% on ParseBench-100 (document/table/layout understanding) vs. 34.6% for the Hermes harness and 13.9% for Pi, using the fewest tokens — [MarkTechPost coverage](https://www.marktechpost.com/2026/08/25/perplexity-ships-portable-computer-on-nvidia-dgx-spark-local-harness-os-enforced-sandbox-and-zero-per-token-cost-for-local-steps/), cross-confirmed by ParseBench's own creator [Jerry Liu / LlamaIndex](https://x.com/jerryjliu0/status/2092642169614200989) that Perplexity used an authentic ParseBench subset, not a self-designed knockoff.
- **Effort:** ~2 hours — pull ParseBench's open methodology (or hand-build a filings-specific analog from a handful of 13F/Form 4 PDFs with known ground truth) and run it against the current in-house filing-extraction path as a baseline.
- **Expected payoff:** Medium — directly on the EDGAR/PDF parsing bottleneck (a named priority area); ParseBench's task design (structure/meaning preservation, not visual similarity) is closer to Darren's actual need than a generic OCR score, but no primary source yet verifies accuracy on SEC-filing-specific documents.
- **Evidence quality:** Medium (Source Medium — vendor benchmark / Evidence Medium — partial mitigation from the ParseBench creator's cross-confirmation, but not independent replication of the 65.1% figure / Interpretation Medium — general document types, not filings-specific; capped).
- **Similar tools already tested:** LlamaIndex's own ParseBench benchmark (same underlying test suite, this week's #6 Also-Considered item, composite 3.0) covers the same bottleneck via a different vendor (LlamaParse) — run one before the other rather than both in the same week to avoid duplicate parsing-benchmark effort.
- **Success metric:** On a 10–20 document sample of 13F/Form 4/merger-proxy PDFs with known ground truth, measured extraction accuracy (table/exhibit fidelity) either exceeds the current in-house parser's baseline, or at minimum establishes a documented baseline where none exists today.
- **Smallest shadow test:** As above — baseline comparison only, no production code change.
- **Ladder move:** primary_source_found → internal_experiment → measured_improvement (if the baseline comparison shows a real accuracy gap).
- **Recommendation: Run** — the only Top-5 item with a real (non-trivial) effort cost, but it's the direct hit on the EDGAR/PDF-parsing bottleneck and produces a documented baseline either way.

---

## Also considered (ranked 6–12, not in Top 5 — all Wait)

6. **LlamaIndex ParseBench data-adapter test** (composite 3.0) — same EDGAR/PDF-parsing bottleneck as Rank 5; run after Rank 5 to avoid duplicating parsing-benchmark effort in the same week. [LlamaIndex](https://www.llamaindex.ai/blog/parsebench)
7. **"Practice Makes Unsafe" skill-misevolution audit vs. Hermes self-improving skill** (composite 3.0) — read-only audit of the HOT/WARM/COLD promotion pipeline against the paper's benchmark methodology; worth doing once PR #19 lands. [arXiv:2608.12851](https://arxiv.org/abs/2608.12851)
8. **Asana/Codex parallel-agent CI-remediation pilot** (composite 2.0) — structurally the closest analog to the #576–#579 CI-red backlog, but requires Darren's explicit sign-off before any agent touches the biotech-screener repo (production lock). [Asana](https://asana.com/inside-asana/migrating-off-enzyme-2-weeks)
9. **AQuA self-improving quant-research sandbox replication** (composite 2.0) — single unreplicated arXiv paper, held at primary_source_found for a third straight week (captured Aug 21, still not advanced as of the Aug 26 retry 2/3). [arXiv:2608.12841](https://arxiv.org/abs/2608.12841)
10. **Codex Harness open-source gap analysis vs. PR #19** (composite 2.0) — useful reference, but PR #19 already has two better-evidenced items queued this week (Rank 3). [OpenAI](https://developers.openai.com/blog/codex-as-a-platform)
11. **GPT-5.6 Terra/Kiro spec-driven-harness CI-cost test** (composite 2.0, merged 3x recapture) — evidence capped Low after 3 failed independent-verification attempts on the 82% figure; also flags the digest's own recurring dedup-logic gap (same URL captured Aug 25, 26, and 27). [OpenAI](https://openai.com/index/gpt-5-6-in-kiro/)
12. **Databricks Unity AI Gateway Smart Routing cost-tiered pilot** (composite 1.0) — evidence held permanently at primary_source_found with no independent confirmation of the 30%+ figure after 3 attempts. [Databricks](https://www.databricks.com/blog/smart-routing-unity-ai-gateway-match-frontier-quality-30-lower-cost-task)

## Dropped this week

- **OpenAI Private Safety Processing (PSP) privacy-governance note** — destination is quarantine watch/context (revisit when the September technical white paper lands), not a build/experiment backlog.
- **Snowflake CoCo Cloud Agents** — captured Aug 22 at ROI 3, but the digest's own next-day reclassification (Aug 23) downgraded it to ROI 2/incremental with no current Wake Robin Snowflake footprint.
- **Cursor `/goal` long-lived agent objective** — same pattern: downgraded to ROI 2/incremental Aug 23; also gated behind the CI-green (20 consecutive clean cycles) precondition regardless.
- **ASI-Bench guided-vs-unguided research-autonomy benchmark** — thematically relevant to research-agent autonomy calibration, but the source note never specifies an expected_roi field; deferred, can't score without it.
- **WebMCP browser-tool-registration primitive** — explicitly "not actionable now" per its own disposition; quarterly monitor only, no action this week.
- **Duplicate recaptures (instrumentation flags only, not separate candidates):** Asana/Enzyme migration re-surfaced Aug 25 (superseded, canonical = Aug 21 / ledger row 132); AQuA re-surfaced Aug 23 (dedup, canonical = Aug 21); OpenAI enterprise ROI null-result re-surfaced Aug 26 via the NBER-style arXiv citation (merged into Rank 1 above); GPT-5.6 Terra/Kiro recaptured three separate times (Aug 25/26/27) — the discovery layer's repeat-recapture pattern is now recurring across at least 5 distinct URLs since mid-August, worth flagging to the digest maintainer as a dedup-logic defect, not a routing/scoring change.

## Next week

- Check whether the Perplexity ParseBench-100 filings test (Rank 5) or the LlamaIndex ParseBench test (Also-Considered #6) actually ran — do not re-recommend both again if one already produced a baseline.
- Check whether PR #19 merged with the four incidents now cited in its review notes (Aug 20's two + this week's two); if merged, this line of recommendation is done and should not recur.
- The GPT-5.6 Terra/Kiro repeat-recapture (5th distinct URL to show this pattern) is worth escalating as a named defect if it recurs an unprecedented 4th time.
- Muse Spark 1.2's security posture (sandbox-escape caution, flagged 8/9) remains unresolved — continue holding any write-path pilot.
- Do not re-recommend the sandbox-escape/write-path cluster (Meta Muse Spark, Kimi K3, OpenAI Astra Critical-tier, AISI pattern) unless a materially new lab/incident emerges beyond what PR #19 already covers.

## Self-check

- Captures reviewed: 7 daily AI digests (Aug 21–27); ~24 discrete items scanned (Aug 24 was a zero-surface day).
- Candidates qualified: 13 (5 Top, 7 Also-Considered/Wait, plus 1 borderline merged into Rank 1).
- Top-N delivered: 5 of 5 requested — no thin-week shortfall this run.
- Destinations written: email (work + user), Content Library, GitHub, Decision Impact Ledger (5 rows) — confirmed below.
- Memory: this week's ledger entry recorded for next week's repeat/supersede check.
