"""Experimento: validação de hiperparâmetros, teste, MLflow, figuras e artefato do modelo."""
import argparse
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datathon.data import (
    PROJECT_ROOT,
    RAW_PATH,
    block_arm_rates,
    clean,
    dataset_sha256,
    evaluation_set,
    load_raw,
)
from datathon.evaluation import (
    ReplayResult,
    best_arm_share,
    block_sums,
    bootstrap_diff_ci,
    effective_sample_size,
    pseudo_regret,
    replay,
    snips,
)
from datathon.features import ARMS, SEGMENTS
from datathon.policies import (
    AlternatingAB,
    DiscountedEpsilonGreedy,
    DiscountedThompsonSampling,
    FixedArm,
)

GAMMAS = (1.0, 0.999, 0.995, 0.99)
EPSILONS = (0.05, 0.1, 0.2)
BASELINE = "ab_deterministico"
MAIN_POLICY = "ts_contextual_desconto"
SEGMENT_DEFINITION = "faixa_etaria(<=30,31-59,>=60) x poutcome==success"
EXPERIMENT_NAME = "datathon-bandit-canal"


@dataclass
class EvalData:
    segments: np.ndarray
    arms: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray
    blocks: np.ndarray
    arm_rates: np.ndarray
    test_mask: np.ndarray

    @classmethod
    def from_frame(cls, ev: pd.DataFrame) -> "EvalData":
        n = len(ev)
        return cls(segments=ev["segment_idx"].to_numpy(), arms=ev["arm"].to_numpy(),
                   rewards=ev["reward"].to_numpy(), propensities=ev["propensity"].to_numpy(),
                   blocks=ev["block"].to_numpy(), arm_rates=block_arm_rates(ev),
                   test_mask=np.arange(n) >= n // 2)

    def head(self, n: int) -> "EvalData":
        return replace(self, segments=self.segments[:n], arms=self.arms[:n],
                       rewards=self.rewards[:n], propensities=self.propensities[:n],
                       blocks=self.blocks[:n], arm_rates=self.arm_rates[:n],
                       test_mask=np.ones(n, dtype=bool))


def run_many(factory, data: EvalData, seeds) -> list[ReplayResult]:
    return [replay(factory(), data.segments, data.arms, data.rewards, np.random.default_rng(s))
            for s in seeds]


def _nan_to_none(value: float):
    return None if isinstance(value, float) and math.isnan(value) else value


def summarize(results: list[ReplayResult], data: EvalData) -> dict:
    m = data.test_mask
    values = [snips(r.matched, data.rewards, data.propensities, m) for r in results]
    by_segment = {}
    for i, seg in enumerate(SEGMENTS):
        seg_mask = m & (data.segments == i)
        seg_vals = [snips(r.matched, data.rewards, data.propensities, seg_mask) for r in results]
        by_segment[seg] = _nan_to_none(float(np.nanmean(seg_vals)) if not all(
            math.isnan(v) for v in seg_vals) else float("nan"))
    return {
        "snips_mean": float(np.mean(values)),
        "snips_std": float(np.std(values)),
        "pseudo_regret_final": float(np.mean([pseudo_regret(r.chosen, data.arm_rates, m)[-1]
                                              for r in results])),
        "best_arm_share": float(np.mean([best_arm_share(r.chosen, data.arm_rates, m)
                                         for r in results])),
        "accepted_events": float(np.mean([(r.matched & m).sum() for r in results])),
        "ess": float(np.mean([effective_sample_size(r.matched, data.propensities, m)
                              for r in results])),
        "snips_by_segment": by_segment,
    }


def select_hyperparameters(val: EvalData, seeds) -> tuple[dict, list[dict]]:
    """Escolhe gamma/epsilon só com os eventos de validação."""
    rows = []

    def score(factory):
        results = run_many(factory, val, seeds)
        return float(np.mean([snips(r.matched, val.rewards, val.propensities) for r in results]))

    ts_scores = {}
    for g in GAMMAS:
        ts_scores[g] = score(lambda g=g: DiscountedThompsonSampling(gamma=g, contextual=True))
        rows.append({"policy": "ts_contextual", "gamma": g, "epsilon": None,
                     "snips_validacao": ts_scores[g]})
    eg_scores = {}
    for e in EPSILONS:
        for g in GAMMAS:
            eg_scores[(e, g)] = score(
                lambda e=e, g=g: DiscountedEpsilonGreedy(epsilon=e, gamma=g, contextual=True))
            rows.append({"policy": "eg_contextual", "gamma": g, "epsilon": e,
                         "snips_validacao": eg_scores[(e, g)]})
    best_eps, best_gamma_eg = max(eg_scores, key=eg_scores.get)
    selected = {"gamma_ts": max(ts_scores, key=ts_scores.get),
                "epsilon_eg": best_eps, "gamma_eg": best_gamma_eg}
    return selected, rows


def policy_specs(selected: dict, best_hist_arm: int) -> dict:
    """nome -> (factory, estocástica?, params)."""
    g, e, ge = selected["gamma_ts"], selected["epsilon_eg"], selected["gamma_eg"]
    return {
        BASELINE: (AlternatingAB, False, {}),
        "sempre_telefone": (lambda: FixedArm(ARMS.index("telephone")), False, {"arm": "telephone"}),
        "melhor_historico": (lambda: FixedArm(best_hist_arm), False, {"arm": ARMS[best_hist_arm]}),
        MAIN_POLICY: (lambda: DiscountedThompsonSampling(gamma=g, contextual=True), True,
                      {"gamma": g, "contextual": True}),
        "ts_contextual_padrao": (lambda: DiscountedThompsonSampling(gamma=1.0, contextual=True),
                                 True, {"gamma": 1.0, "contextual": True}),
        "ts_sem_contexto_desconto": (lambda: DiscountedThompsonSampling(gamma=g, contextual=False),
                                     True, {"gamma": g, "contextual": False}),
        "eg_contextual_desconto": (
            lambda: DiscountedEpsilonGreedy(epsilon=e, gamma=ge, contextual=True), True,
            {"gamma": ge, "epsilon": e, "contextual": True}),
    }


def build_policy_artifact(df: pd.DataFrame, gamma: float, sha: str) -> dict:
    """Posterior com esquecimento sobre o log completo, em ordem cronológica (determinístico)."""
    policy = DiscountedThompsonSampling(gamma=gamma, contextual=True)
    for seg, arm, reward in zip(df["segment_idx"], df["arm"], df["reward"]):
        policy.update(int(seg), int(arm), int(reward))
    segments = {}
    for i, seg in enumerate(SEGMENTS):
        alpha, beta = policy.posterior_params(i)
        segments[seg] = {arm: {"alpha": round(float(alpha[j]), 6), "beta": round(float(beta[j]), 6)}
                         for j, arm in enumerate(ARMS)}
    return {"algorithm": "discounted_thompson_sampling", "gamma": gamma,
            "prior": {"alpha": 1.0, "beta": 1.0}, "arms": list(ARMS), "segments": segments,
            "dataset_sha256": sha, "trained_at": datetime.now(UTC).isoformat()}


def save_figures(summaries: dict, regret_curves: dict, best_curves: dict, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    names = list(summaries)
    paths = []

    fig, ax = plt.subplots(figsize=(9, 4.5))
    means = [summaries[n]["snips_mean"] * 100 for n in names]
    stds = [summaries[n]["snips_std"] * 100 for n in names]
    ax.barh(names, means, xerr=stds, color="#4C72B0")
    ax.axvline(summaries[BASELINE]["snips_mean"] * 100, color="#C44E52", ls="--", label="baseline A/B")
    ax.set_xlabel("Conversão estimada no teste (SNIPS, %)")
    ax.legend()
    fig.tight_layout()
    paths.append(outdir / "conversao_snips.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n, curve in regret_curves.items():
        ax.plot(curve, label=n)
    ax.set_xlabel("Eventos de teste")
    ax.set_ylabel("Pseudo-regret acumulado")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(outdir / "pseudo_regret.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n, curve in best_curves.items():
        ax.plot(curve, label=n)
    ax.set_xlabel("Eventos de teste")
    ax.set_ylabel("% escolhas no melhor canal do bloco (janela 1000)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(outdir / "escolha_melhor_braco.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def _rolling(values: np.ndarray, window: int = 1000) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def log_to_mlflow(tracking_uri: str, common: dict, validation_rows: list[dict], specs: dict,
                  summaries: dict, comparisons: dict, artifacts: list[Path]) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    for row in validation_rows:
        name = f"val_{row['policy']}_g{row['gamma']}" + (f"_e{row['epsilon']}" if row["epsilon"] else "")
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("stage", "validation")
            mlflow.log_params({**common, "policy": row["policy"], "gamma": row["gamma"],
                               "epsilon": row["epsilon"], "split": "validacao"})
            mlflow.log_metric("snips_validacao", row["snips_validacao"])
    for name, summ in summaries.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("stage", "test")
            mlflow.log_params({**common, "policy": name, "split": "teste", **specs[name][2]})
            mlflow.log_metrics({k: v for k, v in summ.items() if isinstance(v, float)})
            for seg, v in summ["snips_by_segment"].items():
                if v is not None:
                    mlflow.log_metric(f"snips_{seg}", v)
            if name in comparisons:
                mlflow.log_metrics({f"vs_baseline_{k}": v for k, v in comparisons[name].items()})
            if name == MAIN_POLICY:
                for path in artifacts:
                    mlflow.log_artifact(str(path))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Roda o experimento do bandit de canal.")
    parser.add_argument("--data", type=Path, default=RAW_PATH)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--n-val-seeds", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "policy.json")
    parser.add_argument(
        "--tracking-uri",
        default=f"sqlite:///{(PROJECT_ROOT / 'mlruns' / 'mlflow.db').as_posix()}",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    sha = dataset_sha256(args.data)
    df = clean(load_raw(args.data))
    ev = evaluation_set(df)
    data = EvalData.from_frame(ev)
    n_val = len(ev) // 2

    selected, validation_rows = select_hyperparameters(data.head(n_val), range(args.n_val_seeds))
    print(f"Hiperparâmetros escolhidos na validação: {selected}")

    first_block = ev[ev["block"] == ev["block"].min()]
    best_hist_arm = int(first_block.groupby("arm")["reward"].mean().idxmax())
    specs = policy_specs(selected, best_hist_arm)

    test_blocks = np.unique(data.blocks[data.test_mask])
    summaries, comparisons, regret_curves, best_curves, sums = {}, {}, {}, {}, {}
    for name, (factory, stochastic, _) in specs.items():
        seeds = range(args.n_seeds) if stochastic else range(1)
        results = run_many(factory, data, seeds)
        summaries[name] = summarize(results, data)
        m = data.test_mask
        regret_curves[name] = np.mean([pseudo_regret(r.chosen, data.arm_rates, m) for r in results], axis=0)
        best = data.arm_rates.argmax(axis=1)
        best_curves[name] = _rolling(np.mean([(r.chosen == best)[m] for r in results], axis=0))
        per_seed = [block_sums(r.matched, data.rewards, data.propensities, data.blocks, test_blocks, m)
                    for r in results]
        sums[name] = (np.array([p[0] for p in per_seed]), np.array([p[1] for p in per_seed]))
        print(f"{name:28s} SNIPS={summaries[name]['snips_mean']:.4f} ± {summaries[name]['snips_std']:.4f}")

    num_b, den_b = sums[BASELINE]
    for name in specs:
        if name != BASELINE:
            num_a, den_a = sums[name]
            comparisons[name] = bootstrap_diff_ci(num_a, den_a, num_b, den_b, n_boot=args.n_boot)

    artifact = build_policy_artifact(df, selected["gamma_ts"], sha)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.model_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "dataset_sha256": sha, "n_rows_clean": len(df), "n_eval_events": len(ev),
        "n_test_events": int(data.test_mask.sum()), "n_seeds": args.n_seeds,
        "n_val_seeds": args.n_val_seeds, "segment_definition": SEGMENT_DEFINITION,
        "best_historical_arm": ARMS[best_hist_arm], "selected": selected,
        "validation": validation_rows, "policies": summaries,
        "comparison_vs_baseline": comparisons,
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    figures = save_figures(summaries, regret_curves, best_curves, args.reports_dir / "figures")

    if not args.no_mlflow:
        common = {"prior_alpha": 1.0, "prior_beta": 1.0, "n_seeds": args.n_seeds,
                  "segment_definition": SEGMENT_DEFINITION, "dataset_sha256": sha,
                  "n_eval_events": len(ev)}
        log_to_mlflow(args.tracking_uri, common, validation_rows, specs, summaries, comparisons,
                      [args.model_path, metrics_path, *figures])
    return metrics


if __name__ == "__main__":
    main()
