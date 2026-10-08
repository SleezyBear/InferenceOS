"""Runtime configuration for InferenceOS.

Configuration is intentionally small in Phase 0: it locates the llama.cpp toolchain,
the model directory, and the artifact directories used for reports/logs/audit data.
Values may be overridden with ``INFERENCEOS_``-prefixed environment variables or a
``.env`` file at the project root.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """Walk up from this file to the directory containing ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Fallback: src/inferenceos/config.py -> project root is three levels up.
    return here.parents[2]


def _rooted(*parts: str) -> Path:
    return find_project_root().joinpath(*parts)


def default_reasoning_model() -> Path | None:
    """The locally available instruct model used as the default reasoner.

    Returns ``None`` when it is not present, so the engine reports itself
    unavailable rather than silently substituting a different model.
    """
    candidate = _rooted("models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    return candidate if candidate.is_file() else None


class Settings(BaseSettings):
    """Resolved, validated configuration for the current machine."""

    model_config = SettingsConfigDict(
        env_prefix="INFERENCEOS_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=find_project_root)
    tools_dir: Path = Field(default_factory=lambda: _rooted("tools"))
    llama_cpp_dir: Path = Field(default_factory=lambda: _rooted("tools", "llama.cpp"))
    models_dir: Path = Field(default_factory=lambda: _rooted("models"))
    reports_dir: Path = Field(default_factory=lambda: _rooted("reports"))
    logs_dir: Path = Field(default_factory=lambda: _rooted("logs"))
    data_dir: Path = Field(default_factory=lambda: _rooted("data"))

    log_level: str = "INFO"

    # Default inference knobs.
    default_gpu_layers: int = 99

    # Phase 1 live telemetry.
    telemetry_interval_s: float = 0.6
    telemetry_top_processes: int = 8
    # Full process-table scans (psutil ``memory_info`` for every process) are expensive
    # on Windows (~0.2-0.5 s for ~200 processes). The runtime sampler therefore refreshes
    # its process snapshot at this cadence and reuses it between refreshes; the target
    # process is always refreshed cheaply. This reduces per-sample cost without removing
    # the process information diagnosis relies on.
    telemetry_process_scan_interval_s: float = 15.0
    # Runtime GPU telemetry reuses the existing Windows GPU Engine counter sampler at a
    # low, monitoring-mode-controlled cadence (never per-tick). Failure is represented as
    # unavailable, never fabricated.
    telemetry_runtime_gpu_enabled: bool = True
    telemetry_gpu_scan_interval_s: float = 5.0
    # Runtime paging telemetry reuses the same Windows ``\Memory\Pages Input/sec``
    # counter sampler the Phase 3 harness uses, at a low cadence. Unavailable telemetry
    # is explicit (never a fabricated 0.0); a genuine measured zero is distinguishable.
    telemetry_runtime_paging_enabled: bool = True
    telemetry_paging_scan_interval_s: float = 5.0

    # Phase 6 candidate generation / controlled optimization policy.
    phase6_max_candidates: int = 8
    phase6_min_diagnosis_confidence: float = 0.0
    phase6_min_evidence_strength: str = "low"
    phase6_context_reduction_fraction: float = 0.5
    phase6_min_context_length: int = 512
    phase6_autonomous_max_risk: str = "medium"
    phase6_allow_reasoning: bool = True
    phase6_max_optimizer_overhead_s: float = 120.0
    phase6_experiment_repetitions: int = 4
    phase6_weight_benefit: float = 0.30
    phase6_weight_uncertainty: float = 0.08
    phase6_weight_cost: float = 0.15
    phase6_weight_overhead: float = 0.05
    phase6_weight_risk: float = 0.15
    phase6_weight_reversibility: float = 0.07
    phase6_weight_confidence: float = 0.08
    phase6_weight_causal_strength: float = 0.06
    phase6_weight_scope: float = 0.03
    phase6_weight_approval: float = 0.02
    phase6_weight_precondition: float = 0.01

    # Resource-aware multi-candidate search (GPU-offload layers first). These knobs
    # drive the feasible-range construction, deterministic sampling and selection.
    # The safety headroom is an explicit fraction of total VRAM that is never
    # allocated to model weights; the runtime reserve covers runtime/workspace
    # allocations. Telemetry older than the max age is treated as stale (never
    # fabricated fresh).
    phase6_gpu_offload_safety_headroom_fraction: float = 0.15
    phase6_gpu_offload_runtime_reserve_fraction: float = 0.05
    phase6_gpu_offload_telemetry_max_age_s: float = 30.0
    phase6_gpu_offload_sample_step_layers: int = 2
    phase6_gpu_offload_max_samples: int = 6
    phase6_gpu_offload_probe_step_layers: int = 2
    phase6_gpu_offload_diminishing_returns_margin: float = 0.05
    phase6_gpu_offload_kv_element_bytes: int = 2
    # Measurement robustness (candidate measurement validity only; these do NOT change
    # the search space, objective, decision thresholds or resource model).
    # ``max_measure_attempts`` bounds retries of an *invalid/contaminated* attempt
    # (initial attempt + 2 retries by default). ``max_measure_cv`` is the primary-metric
    # coefficient of variation above which an attempt is treated as contaminated (it
    # defaults to the Phase 7 noise tolerance value). ``paging_per_sec_ceiling`` is an
    # optional measurement-validity paging ceiling (None disables it).
    phase6_gpu_offload_max_measure_attempts: int = 3
    phase6_gpu_offload_max_measure_cv: float = 0.10
    phase6_gpu_offload_paging_per_sec_ceiling: float | None = None

    # Phase 7 experiment execution / quality validation policy.
    phase7_warmup_repetitions: int = 1
    phase7_measured_repetitions: int = 3
    phase7_max_repetitions: int = 5
    phase7_min_valid_samples: int = 2
    phase7_max_total_runs: int = 12
    phase7_time_limit_s: float = 900.0
    phase7_run_timeout_s: float = 300.0
    phase7_min_effect: float = 0.05
    phase7_noise_tolerance: float = 0.10
    phase7_quality_similarity_min: float = 0.60
    phase7_require_quality_validation: bool = True
    phase7_restore_baseline: bool = True

    # Phase 7R quality-validation suite + adaptive tolerance.
    phase7r_suite_version: str = "7r.1"
    phase7r_base_similarity: float = 0.60
    phase7r_content_overlap_min: float = 0.30
    phase7r_length_ratio_min: float = 0.40
    phase7r_length_ratio_max: float = 2.50
    phase7r_runtime_only_relaxation: float = 0.10
    phase7r_semantic_tightening: float = 0.05
    phase7r_min_similarity: float = 0.40
    phase7r_max_similarity: float = 0.95

    # Phase 8 persistent knowledge & rollback.
    phase8_db_path: Path = Field(default_factory=lambda: _rooted("data", "knowledge.db"))
    phase8_lookup_limit: int = 50
    phase8_verify_integrity: bool = True
    phase8_retention_days: float = 0.0
    phase8_revalidation_max_age_days: float = 30.0
    phase8_condition_tolerance: float = 0.25
    phase8_not_applicable_critical_threshold: int = 2

    # Phase 9 pre-deployment autonomous optimization loop.
    phase9_max_iterations: int = 4
    phase9_max_experiments: int = 4
    phase9_max_wall_clock_s: float = 1800.0
    phase9_max_optimizer_overhead_s: float = 300.0
    phase9_max_deep_diagnoses: int = 2
    phase9_allow_deep_diagnosis: bool = True
    phase9_require_reasoning_when_escalated: bool = False
    phase9_require_knowledge: bool = False
    phase9_knowledge_lookup_limit: int = 20
    phase9_skip_directly_negative: bool = True
    phase9_allow_revalidation: bool = True
    phase9_max_candidates_per_iteration: int = 3
    phase9_autonomous_only: bool = True
    phase9_stop_on_quality_regression: bool = True
    phase9_stop_on_restoration_failure: bool = True
    phase9_reobserve_after_retention: bool = True
    phase9_knowledge_positive_bonus: float = 0.05
    phase9_knowledge_negative_penalty: float = 0.15
    phase9_min_experiment_confidence: float = 0.0

    # Phase 10 runtime adaptive monitoring (conservative mode).
    phase10_telemetry_interval_s: float = 0.5
    phase10_observation_window_size: int = 120
    phase10_baseline_samples: int = 8
    # Workload-aware baseline establishment: the runtime baseline is only eligible once
    # this many baseline observations carrying real workload-performance metrics
    # (latency/TTFT/decode throughput) have been observed, bounded by
    # ``phase10_baseline_max_wait_s`` so a stalled workload cannot hang monitoring.
    phase10_baseline_workload_samples: int = 2
    phase10_baseline_max_wait_s: float = 60.0
    # Adaptive monitoring intensity (separate from monitor lifetime). Excessive measured
    # monitoring overhead reduces sampling intensity (LOW_COST), it does not terminate
    # the monitor. DORMANT engages after this much target inactivity.
    phase10_monitor_overhead_fraction_high: float = 0.20
    phase10_monitor_overhead_fraction_low: float = 0.05
    phase10_monitor_active_window_s: float = 15.0
    phase10_monitor_dormant_after_s: float = 120.0
    phase10_monitor_low_cost_interval_s: float = 2.0
    phase10_monitor_dormant_interval_s: float = 5.0
    phase10_monitor_expensive_scan_interval_s: float = 15.0
    phase10_monitor_low_cost_scan_interval_s: float = 60.0
    phase10_monitor_dormant_scan_interval_s: float = 300.0
    phase10_monitor_gpu_scan_interval_s: float = 5.0
    phase10_opportunity_resuppress_score_delta: float = 0.15
    phase10_opportunity_resuppress_cooldown_s: float = 120.0
    phase10_max_duration_s: float = 900.0
    phase10_max_deep_analyses: int = 1
    phase10_max_interventions: int = 2
    phase10_max_cumulative_intervention_s: float = 300.0
    phase10_max_optimizer_overhead_s: float = 120.0
    phase10_approval_timeout_s: float = 60.0
    phase10_max_alerts_per_condition: int = 2
    phase10_persistence_samples: int = 3
    phase10_condition_lookback_samples: int = 6
    phase10_deviation_threshold: float = 0.25
    phase10_hysteresis_ratio: float = 0.5
    phase10_baseline_drift_tolerance: float = 0.15
    phase10_diagnosis_cooldown_s: float = 30.0
    phase10_intervention_cooldown_s: float = 120.0
    phase10_min_evidence_strength: str = "low"
    phase10_require_approval_for_disruptive: bool = True
    phase10_allow_non_disruptive_auto_adaptation: bool = False
    phase10_max_observation_processes: int = 6
    phase10_benefit_scale_s: float = 60.0
    phase10_min_net_value: float = 0.05
    phase10_deep_analysis_evidence_threshold: float = 0.45

    # Phase 10 search integration: when a GPU-offload opportunity is diagnosed, the
    # monitor runs the resource-aware multi-point search against the live resident
    # workload before requesting approval or applying anything. The search is bounded
    # by its own time limit and reuses the optimization/search policy for sampling and
    # resource-envelope construction. ``phase10_gpu_search_enabled`` allows the search
    # to be disabled (falling back to the legacy single-candidate path) without changing
    # any other monitoring behaviour.
    phase10_gpu_search_enabled: bool = True
    phase10_gpu_search_time_limit_s: float = 240.0
    phase10_gpu_search_paging_per_sec_ceiling: float | None = None

    # Pre-experiment economic affordability for unknown-benefit interventions. This is a
    # *permission* budget (seconds), not a benefit claim: a cheap, reversible candidate
    # whose candidate-specific estimated cost is within the mode-scaled budget may be
    # attempted; its economic worth is decided AFTER measurement. See also
    # ``optimization_*`` mode settings below.
    optimization_experiment_cost_tolerance_s: float = 60.0

    # User-controlled optimization preference (program-level, persisted via env/.env).
    # This only changes *economics* (how much improvement is worth a given search cost)
    # and the bounded *search budget*; it never overrides a hard safety gate.
    optimization_mode: str = "balanced"
    # Custom-mode values (used only when ``optimization_mode == "custom"``).
    optimization_minimum_improvement_percent: float | None = None
    optimization_maximum_search_time_s: float | None = None
    optimization_custom_degraded_minimum_improvement_percent: float | None = None
    optimization_custom_severe_minimum_improvement_percent: float | None = None
    optimization_custom_degraded_search_time_multiplier: float = 1.0
    optimization_custom_severe_search_time_multiplier: float = 1.0
    # Whether degraded performance lowers the required improvement (adaptive urgency).
    optimization_degraded_performance_policy: str = "lower_requirement"
    # Performance-state thresholds (relative workload-performance regression vs baseline).
    optimization_degraded_threshold: float = 0.25
    optimization_severe_degraded_threshold: float = 0.50
    # Hard upper bound on any search budget; there is no unlimited search.
    optimization_max_search_time_hard_cap_s: float = 1200.0
    # Bounded resume verification after a (potentially long) search: the timeout is
    # base + a capped search-duration allowance, always within the hard maximum.
    optimization_resume_base_timeout_s: float = 30.0
    optimization_resume_search_factor: float = 0.25
    optimization_resume_max_search_allowance_s: float = 120.0
    optimization_resume_hard_max_s: float = 300.0

    # Effort mode (step 4). ``effort_mode`` is the new name for ``optimization_mode``;
    # when it is set it wins, otherwise ``optimization_mode`` is used, so existing
    # settings files keep working.
    effort_mode: str | None = None
    # Time limit for the search before a model loads. ``None`` uses the effort mode's
    # own limit (balanced 400 s, performance first 600 s, minimize cost 200 s).
    optimization_before_loading_search_time_s: float | None = None

    # Goal mode (step 4): what counts as better. ``balanced`` or ``fastest_answers``,
    # or ``custom`` with ``goal_custom_measures`` listing the measures in order of
    # importance, comma separated, from: writing_speed, first_word, prompt_reading,
    # answer_time, memory, paging.
    goal_mode: str = "balanced"
    goal_custom_measures: str = ""
    # How much a guarded measure may get worse before a change is refused (percent).
    goal_guard_limit_percent: float = 10.0

    # Payback check (step 4): how much answering time a setup is assumed to get when
    # deciding whether a change is worth it. A setup found before loading is reused
    # for a long time; a change found while running only helps the rest of the session.
    payback_answering_hours_before_loading: float = 10.0
    payback_answering_hours_while_running: float = 1.0

    # Step 5: the everyday "tune" (search, approve, save) and "start" commands.
    # Built-in test prompt used for every measurement of a tune run.
    tune_workload: str = "short-short"
    # No longer used by ``tune`` (4 Oct 2026: the final and re-checks use the fair
    # comparison's planned looks); kept so old settings files still load.
    tune_confirm_rounds: int = 3
    tune_recheck_rounds: int = 2
    # Share of the effort mode's search time kept back for the confirmation.
    tune_confirm_share: float = 0.25
    # A measurement is "disturbed" (and repeated) when other activity used more than
    # this share of the processor during it.
    tune_disturbance_cpu_percent: float = 25.0
    tune_max_disturbed_repeats: int = 2
    # How long to wait for a yes or no on each proposed change.
    tune_approval_timeout_s: float = 300.0
    # Layer search: exactly this many measured runs per layer test (no early stop).
    tune_layer_runs: int = 5
    # Layer search: within this margin the smaller layer count wins the tie.
    tune_layer_tie_margin: float = 0.005
    # Fair comparison (tune.fair): the current and the new setup are measured in turns
    # this many times each; a measure counts as changed only when every turn agrees on
    # the direction and the middle change is at least the noise floor.
    tune_fair_turns: int = 3
    tune_fair_noise_floor: float = 0.01
    # Fair comparison in independent blocks (3 Oct 2026, user decisions): a change that
    # does not really help may be called better at most ``tune_fair_alpha`` of the time;
    # a "must not lose" check passes only when any loss is shown to be smaller than
    # ``tune_guard_margin``. The blocks are looked at after ``tune_fair_min_blocks`` and
    # (when not clear yet) after ``tune_fair_max_blocks``; set from the measured noise.
    # 4 Oct 2026 (user decisions after review 2): the "must not lose" margin is 2% in
    # every effort mode; looks by effort mode (``tune/fair.py`` ``planned_looks``):
    # Minimize cost 4, Balanced/Custom 4 and 8, Performance first 4 and
    # ``tune_fair_max_blocks_performance`` (at most 16; 16 from the laptop noise: a 2%
    # gain needs 9-15 blocks on short questions, and Performance first's smallest
    # worthwhile gain is 1.5%). Unusable blocks are re-attempted at most
    # ``tune_fair_extra_attempts`` times in total.
    tune_fair_alpha: float = 0.05
    tune_guard_margin: float = 0.02
    tune_fair_min_blocks: int = 4
    tune_fair_max_blocks: int = 8
    tune_fair_max_blocks_performance: int = 16
    tune_fair_extra_attempts: int = 2
    # Measured runs per measurement in the paths and fair checks (no early stop). Per
    # question, tune raises it (up to ``tune_path_max_runs``) until the goal score's
    # run-to-run spread is about ``tune_run_noise_target`` of a measurement (4 Oct 2026:
    # short questions are noisy between runs, not between loads).
    tune_path_runs: int = 5
    tune_path_max_runs: int = 20
    # No longer used (replaced by the certainty-per-minute rule, 4 Oct 2026).
    tune_run_noise_target: float = 0.01
    # Load-to-load spread of the goal score (laptop, 4 Oct 2026: about 1-1.5% on the
    # Gemma and 9B short questions with many runs); used to choose runs per measurement.
    tune_between_load_spread: float = 0.012
    # Diagnosis first (4 Oct 2026): before the paths, and again after each kept change,
    # look at where one written word's time goes and order the paths by it.
    tune_diagnosis: bool = True
    # Engine-switch correctness (reviews 4 and 5): the same words in every answer, and
    # each word's probability with and without a switch within this much (provisional).
    tune_switch_prob_tolerance: float = 0.03
    # Looking again at finished paths after a kept change (owner, 4 Oct 2026): only the
    # paths the change can affect, at most this many extra attempts per path, within
    # the remaining time.
    tune_path_revisits: int = 2
    # Block-kind grouped placement for hybrid models (reviewer, 6 Oct 2026): generated
    # and measured, but only accepted once a hardware probe has confirmed it runs
    # correctly on this engine. Set to true after that probe.
    tune_grouped_placement_probed: bool = False
    # Internal experiment only (handoff, 6 Oct 2026; soft target after review 8): a
    # practical time *target* in seconds for the whole tune with the planner's search
    # order (``tune/budgeted.py``). Exploration stops when it would push the final check
    # past the target; the final check may finish up to the extension below past it.
    # 0 = off (the normal tune, unchanged). Not a product mode; no default changes.
    tune_experimental_budget_s: float = 0.0
    # Largest overrun allowed to finish a started measurement pair or the final check:
    # this share of the target, but at least ``..._extension_min_s``.
    tune_experimental_extension_share: float = 0.2
    tune_experimental_extension_min_s: float = 120.0
    # An operation is treated as stuck when it runs this many times longer than its
    # measured expected time (and at least the grace seconds longer).
    tune_experimental_stuck_factor: float = 3.0
    tune_experimental_stuck_grace_s: float = 120.0
    # Weak readings re-measured (best first) once the clear gains are exhausted.
    tune_experimental_weak_repeats: int = 2
    # Fewest blocks of the budgeted search's single final look per question (5%
    # false-keep); more (up to 16) when this tune's noise needs them for about 80% power.
    tune_experimental_final_blocks: int = 4
    # Seconds kept back for approval, saving and stopping the engine.
    tune_experimental_save_reserve_s: float = 30.0
    # Experimental (review 10), off by default: one counter monitor per measurement
    # (warm-up and scored answers), each answer read from its own time window, instead
    # of a monitor started and stopped for every answer. Scoring is unchanged.
    tune_experimental_measurement_monitor: bool = False
    # How strictly each path checks its best candidate before the final check (owner,
    # 4 Oct 2026: keep one fresh final check, cut repeated checking before it, and
    # prove the whole method by simulation first; ``scripts/simulate_tune.py``):
    # Owner's choice after the simulation (4 Oct 2026): "light".
    # "full" = the normal fair check (looks and false-keep rate of the effort mode);
    # "light" = one look at ``tune_path_check_blocks`` blocks with false-keep rate
    # ``tune_path_check_alpha``; "none" = the best first look goes straight on.
    # The final check of the whole setup is always the full one.
    tune_path_check: str = "light"
    tune_path_check_alpha: float = 0.2
    tune_path_check_blocks: int = 4
    # A best first look below this goes to no fair check (0 = any gain is checked).
    tune_path_check_min_first_look: float = 0.0
    # Balanced's final "must not lose" checks: looks at these block counts (owner,
    # 4 Oct 2026, after the simulation in ``scripts/simulate_tune.py``).
    tune_final_guard_looks_balanced: str = "8,16"
    # Trade-offs (owner, 5 Oct 2026): a change clearly better on one question but losing
    # more than the allowed 2% on another is never kept on its own (not even with
    # --yes); the best one is measured properly at the end and offered for a yes or no.
    tune_offer_tradeoffs: bool = True
    # Processor-speed changes larger than this many percentage points during a fair
    # comparison are reported (never used to leave a block out, 4 Oct 2026).
    tune_fair_clock_tolerance: float = 8.0
    # Heat check at the end of tune: the chosen setup answers this many questions in a
    # row (0 = off) with this workload, and any slowdown from heat is reported.
    tune_heat_check_runs: int = 8
    tune_heat_check_workload: str = "short-long-decode"
    # No longer used (4 Oct 2026): the layer path's "chosen vs start" check always uses
    # the fair comparison (one acceptance route); kept so old settings files still load.
    tune_layer_fair_check: bool = True
    # Run the graphics card layer path at all (off: keep the starting layer count).
    tune_layer_path: bool = True
    # Optimization paths run after the graphics card layer path, in this order.
    tune_paths: str = (
        "threads,reading_threads,flash_attention,prompt_batch,engine_switches,placement,kv_cache"
    )
    # The final whole-setup check against the starting setup (turns each).
    tune_final_check_turns: int = 2
    # The long test question (paths that change how a question is read). Its length is
    # up to half the context, cut so one answer takes at most this many seconds on the
    # model being tuned (Balanced, Minimize cost, Custom; Performance first for now).
    tune_long_question_seconds: float = 20.0
    # A fixed long-question length in pieces (0 = fit to the time above). Capped at
    # half the context. Used to compare question lengths.
    tune_long_question_tokens: int = 0
    # Paths that try values around a best one try its neighbours (or one step further
    # at the edge of the range) at most this many extra times.
    tune_path_refine_steps: int = 2
    # Where `start` serves the model (this computer only).
    start_port: int = 8080

    # Phase 11 external process recommendations (staged, minimal; recommendation-only
    # by default, external actions require explicit contextual approval).
    phase11_observation_interval_s: float = 1.0
    phase11_scope: str = "external_process"
    phase11_max_duration_s: float = 300.0
    phase11_max_processes: int = 40
    phase11_max_recommendations: int = 4
    phase11_max_actions: int = 1
    phase11_max_cumulative_action_s: float = 60.0
    phase11_memory_pressure_threshold: float = 0.75
    phase11_memory_dominance_threshold: float = 0.6
    phase11_cpu_dominance_threshold: float = 0.5
    phase11_min_relative_significance: float = 0.5
    phase11_min_process_rss_bytes: int = 268435456
    phase11_min_process_cpu_percent: float = 25.0
    phase11_require_target_degradation: bool = True
    phase11_require_approval_for_actions: bool = True
    phase11_restore_after_action: bool = True
    phase11_elevation_never_automatic: bool = True
    phase11_allow_unsupported_actions: bool = False
    phase11_cooldown_s: float = 300.0
    phase11_approval_timeout_s: float = 60.0
    phase11_max_alerts_per_identity: int = 2
    phase11_benefit_scale_s: float = 120.0
    phase11_min_net_value: float = 0.0
    phase11_db_path: Path = Field(default_factory=lambda: _rooted("data", "external_knowledge.db"))

    # B1 measurement-validity: power/thermal/frequency confound protection.
    # A sustained CPU performance ratio below the floor, corroborated by battery power
    # or a material drop relative to the best state observed in the same session, marks
    # the measurement confounded. ``unknown_blocks`` fails closed when telemetry is
    # unavailable; default off keeps the system usable without power telemetry.
    phase_validity_cpu_ratio_floor: float = 0.60
    phase_validity_cpu_ratio_deviation: float = 0.25
    phase_validity_persistence_samples: int = 2
    phase_validity_unknown_blocks: bool = False

    # Phase 5R resident backend & deep-diagnosis lifecycle.
    resident_host: str = "127.0.0.1"
    resident_port: int = 0  # 0 = dynamically select a free loopback port
    resident_readiness_timeout_s: float = 180.0
    resident_request_timeout_s: float = 600.0
    phase5r_reasoner_expected_bytes: int | None = None
    phase5r_feasibility_headroom_fraction: float = 0.15
    phase5r_reasoner_reclaim_fraction: float = 0.9
    phase5r_benefit_uncertainty_weight: float = 1.0
    phase5r_benefit_competing_weight: float = 0.5
    phase5r_benefit_scale_s: float = 300.0
    phase5r_cost_weight: float = 1.0
    phase5r_min_net_utility: float = 0.0
    phase5r_diagnosis_estimate_s: float = 20.0
    phase5r_reasoner_load_estimate_s: float = 8.0
    phase5r_reasoner_unload_estimate_s: float = 1.0
    phase5r_corroboration_ceiling: float = 0.25
    phase5r_transition_log_limit: int = 200

    # B2 resident-server crash detection & bounded recovery. Recovery is explicitly
    # bounded to a single controlled attempt per failure event; when it cannot be
    # verified the target is marked unrecoverable and the operation fails closed.
    resident_recovery_enabled: bool = True

    # H2 log/report retention & disk protection for unattended operation.
    # Disposable artifacts (allowlisted generated reports under ``reports_dir`` and the
    # per-phase operational logs) are bounded; durable state (SQLite databases,
    # deployment/rollback state, configuration) is never automatically deleted.
    retention_enabled: bool = True
    retention_report_max_files: int = 1000
    retention_report_max_bytes: int = 512 * 1024 * 1024
    retention_max_deletions_per_pass: int = 500
    retention_min_free_bytes: int = 1024 * 1024 * 1024
    log_max_bytes: int = 8 * 1024 * 1024
    log_backup_count: int = 3

    # Phase 5 local reasoning (deep-diagnosis mode).
    reasoning_enabled: bool = True
    reasoning_provider: str = "llama_cpp"
    reasoning_model_path: Path | None = Field(default_factory=default_reasoning_model)
    reasoning_cli_path: Path | None = None
    reasoning_context_length: int = 8192
    reasoning_gpu_layers: int = 99
    reasoning_max_tokens: int = 1024
    reasoning_temperature: float = 0.0
    reasoning_seed: int = 1234
    reasoning_timeout_s: float = 300.0
    reasoning_escalation_threshold: float = 0.6
    reasoning_max_evidence: int = 12

    def ensure_directories(self) -> None:
        """Create the artifact directories if they do not yet exist."""
        for directory in (
            self.tools_dir,
            self.llama_cpp_dir,
            self.models_dir,
            self.reports_dir,
            self.logs_dir,
            self.data_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
