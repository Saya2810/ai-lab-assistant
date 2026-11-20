"""Turn benchmark results into a self-contained HTML report with plots.

The plots follow what the IR / RAG evaluation literature uses (references in the report):

* Hit@k curve and Hit@1/Hit@3/MRR/nDCG@10 with bootstrap confidence intervals
  (BEIR/MTEB report nDCG@10 and Recall@k);
* ablation as paired differences per question with bootstrap CI and a paired
  randomisation test (recommended for IR by Smucker, Allan & Carterette 2007);
* similarity distributions of relevant vs irrelevant chunks with the thresholds;
* ROC curve (AUROC) and risk-coverage curve for "is it in the notebook at all"
  (selective prediction, Geifman & El-Yaniv 2017);
* confusion matrix of the planner's routing; latency per step and its ECDF;
* a per-question heatmap to find the failing questions;
* for the Lab Assistant: measurement precision/recall/F1, invented numbers, latency.

Needs matplotlib (``pip install -e ".[eval]"``); everything else is in the result JSON.
"""

from __future__ import annotations

import base64
import html
import io
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from lab_chatbot.benchmark import auroc, bootstrap_ci

COLORS = ["#2f6db5", "#d9822b", "#3a9a5b", "#b5443a", "#7b5ea7", "#7f7f7f"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    _plt().close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _rr(rank: int | None) -> float:
    return 1 / rank if rank else 0.0


def _answerable(questions: list[dict]) -> list[dict]:
    return [q for q in questions if q["expect_ids"] and not q.get("error")]


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def plot_hit_at_k(result: dict) -> str:
    plt = _plt()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    ks = list(range(1, 11))
    for i, (name, data) in enumerate(result["sets"].items()):
        curve = data["retrieval"].get("hit_at")
        if curve:
            left.plot(ks, [curve[k] if k in curve else curve[str(k)] for k in ks], "o-", ms=3,
                      color=COLORS[i % 6], label=f"{name} (n={data['retrieval']['n']})")  # fmt: skip
    left.set(title="Hit@k je Fragensatz", xlabel="k (Anzahl angezeigter Experimente)", ylabel="Anteil Fragen", ylim=(0, 1.05))
    left.legend(frameon=False)
    questions = _answerable(result["questions"])
    for i, variant in enumerate(v["name"] for v in result["variants"]):
        ranks = [q["variants"][variant]["rank"] for q in questions]
        right.plot(ks, [sum(1 for r in ranks if r and r <= k) / max(1, len(ranks)) for k in ks], "o-", ms=3,
                   color=COLORS[i % 6], label=variant)  # fmt: skip
    right.set(title=f"Hit@k je Suchvariante (alle beantwortbaren, n={len(questions)})", xlabel="k")
    right.legend(frameon=False)
    return _png(fig)


def metric_rows(questions: list[dict], variant: str | None = None) -> dict[str, list[float]]:
    def rank(q):
        return q["variants"][variant]["rank"] if variant else q["rank"]

    return {
        "Hit@1": [float(rank(q) == 1) for q in questions],
        "Hit@3": [float(bool(rank(q)) and rank(q) <= 3) for q in questions],
        "MRR": [_rr(rank(q)) for q in questions],
    }


def plot_metrics_ci(result: dict) -> tuple[str, list[list]]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3))
    sets = [(n, [q for q in _answerable(result["questions"]) if q["set"] == n]) for n in result["sets"]]
    sets = [(n, qs) for n, qs in sets if qs]
    names = ["Hit@1", "Hit@3", "MRR"]
    width = 0.8 / max(1, len(sets))
    table = []
    for i, (set_name, qs) in enumerate(sets):
        rows = metric_rows(qs)
        means = [float(np.mean(rows[m])) for m in names]
        cis = [bootstrap_ci(rows[m]) for m in names]
        x = np.arange(len(names)) + i * width
        ax.bar(x, means, width, color=COLORS[i % 6], label=f"{set_name} (n={len(qs)})",
               yerr=[[m - lo for m, (lo, _) in zip(means, cis)], [hi - m for m, (_, hi) in zip(means, cis)]],
               capsize=3)  # fmt: skip
        ndcg = result["sets"][set_name]["retrieval"].get("ndcg10")
        table.append([set_name, len(qs), *(f"{m:.2f} [{lo:.2f}, {hi:.2f}]" for m, (lo, hi) in zip(means, cis)),
                      f"{ndcg:.2f}" if ndcg is not None else "–"])  # fmt: skip
    ax.set_xticks(np.arange(len(names)) + width * (len(sets) - 1) / 2, names)
    ax.set(ylim=(0, 1.1), title="Suchqualität mit 95 %-Bootstrap-Konfidenzintervall")
    ax.legend(frameon=False, fontsize=8)
    return _png(fig), table


def paired_permutation_p(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 11) -> float:
    """Two-sided paired randomisation test on the mean difference."""
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(float(np.mean(diffs))) if diffs else 0.0
    if not diffs or observed == 0:
        return 1.0
    rng = random.Random(seed)
    hits = sum(abs(np.mean([d if rng.random() < 0.5 else -d for d in diffs])) >= observed - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def plot_ablation(result: dict) -> tuple[str, list[list]]:
    plt = _plt()
    questions = _answerable(result["questions"])
    base = [_rr(q["variants"]["hybrid"]["rank"]) for q in questions]
    rows, table = [], []
    for variant in [v["name"] for v in result["variants"] if v["name"] != "hybrid"]:
        other = [_rr(q["variants"][variant]["rank"]) for q in questions]
        diffs = [o - b for o, b in zip(other, base)]
        lo, hi = bootstrap_ci(diffs)
        p = paired_permutation_p(other, base)
        rates = [s["false_answers"][variant] for s in result["sets"].values() if s["false_answers"]]
        rows.append((variant, float(np.mean(diffs)), lo, hi))
        table.append([variant, f"{np.mean(diffs):+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{p:.3f}",
                      f"{np.mean(rates):.0%}" if rates else "–"])  # fmt: skip
    hybrid_false = [s["false_answers"].get("hybrid") for s in result["sets"].values() if s["false_answers"]]
    table.insert(0, ["hybrid (Referenz)", "0", "–", f"{np.mean(hybrid_false):.0%}" if hybrid_false else "–"])
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * max(1, len(rows))))
    for i, (name, mean, lo, hi) in enumerate(rows):
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", color=COLORS[3] if hi < 0 else COLORS[0], capsize=3)
    ax.axvline(0, color="#999", lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set(xlabel="Differenz im Reciprocal Rank gegenüber dem vollen System (pro Frage gepaart)",
           title=f"Ablation: was bringt welcher Baustein? (n={len(questions)} Fragen)")  # fmt: skip
    return _png(fig), table


def plot_scores(result: dict) -> str:
    plt = _plt()
    rel = [s for q in _answerable(result["questions"]) for s in q.get("scores_relevant", [])]
    irr = [s for q in _answerable(result["questions"]) for s in q.get("scores_irrelevant", [])]
    neg = [s for q in result["questions"] if q["unanswerable"] for s in q.get("scores_irrelevant", [])]
    fig, ax = plt.subplots(figsize=(7, 3))
    bins = np.linspace(-0.1, 0.8, 37)
    ax.hist(irr, bins=bins, alpha=0.55, color=COLORS[5], label=f"Abschnitte anderer Experimente (n={len(irr)})", density=True)
    if neg:
        ax.hist(neg, bins=bins, alpha=0.45, color=COLORS[3], label=f"bei Fragen ohne Antwort im Laborbuch (n={len(neg)})", density=True)
    ax.hist(rel, bins=bins, alpha=0.65, color=COLORS[2], label=f"Abschnitte des richtigen Experiments (n={len(rel)})", density=True)
    threshold = result["config"]["min_dense"]
    for value, style, label in ((threshold, "-", "Schwelle nur Bedeutung"), (threshold - 0.05, "--", "nur über Synonyme"),
                                (threshold - 0.10, ":", "mit Stichwort der Frage")):  # fmt: skip
        ax.axvline(value, color="#333", ls=style, lw=1, label=f"{label} ({value:.2f})")
    ax.set(xlabel="Kosinus-Ähnlichkeit zur Frage (bessere der zwei Formulierungen)", ylabel="Dichte",
           title="Wie gut trennt die Vektorsuche richtig von falsch?")  # fmt: skip
    ax.legend(frameon=False, fontsize=7)
    return _png(fig)


def plot_abstention(result: dict) -> tuple[str, dict]:
    plt = _plt()
    questions = [q for q in result["questions"] if not q.get("error") and (q["expect_ids"] or q["unanswerable"])]
    pos = [q["max_score"] for q in questions if q["expect_ids"]]
    neg = [q["max_score"] for q in questions if not q["expect_ids"]]
    fig, (roc_ax, rc_ax) = plt.subplots(1, 2, figsize=(9, 3.4))
    thresholds = sorted({*pos, *neg, 2.0}, reverse=True)
    tpr = [sum(p >= t for p in pos) / max(1, len(pos)) for t in thresholds]
    fpr = [sum(n >= t for n in neg) / max(1, len(neg)) for t in thresholds]
    area = auroc(pos, neg)
    roc_ax.plot([0, *fpr, 1], [0, *tpr, 1], color=COLORS[0], label=f"AUROC = {area:.2f}" if area is not None else "")
    roc_ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1)
    answered_pos = sum(bool(q["ranked"]) for q in questions if q["expect_ids"]) / max(1, len(pos))
    answered_neg = sum(bool(q["ranked"]) for q in questions if not q["expect_ids"]) / max(1, len(neg))
    roc_ax.plot([answered_neg], [answered_pos], "o", color=COLORS[3], label="aktuelles System")
    roc_ax.set(xlabel="Anteil Fragen ohne Antwort, die trotzdem Treffer bekommen", ylabel="Anteil beantwortbarer mit Treffern",
               title=f"Steht es im Laborbuch? (ROC, {len(pos)} vs {len(neg)} Fragen)")  # fmt: skip
    roc_ax.legend(frameon=False, loc="lower right")

    # risk-coverage: answer the most confident questions first
    order = sorted(questions, key=lambda q: -q["max_score"])
    correct = [bool(q["expect_ids"]) and q["rank"] is not None and q["rank"] <= 3 for q in order]
    coverage = [i / len(order) for i in range(1, len(order) + 1)]
    risk = [1 - sum(correct[:i]) / i for i in range(1, len(order) + 1)]
    aurc = float(np.trapezoid(risk, coverage)) if len(order) > 1 else float("nan")
    rc_ax.plot(coverage, risk, color=COLORS[1], label=f"AURC = {aurc:.2f} (kleiner ist besser)")
    rc_ax.set(xlabel="Abdeckung (Anteil beantworteter Fragen)", ylabel="Fehlerrate unter den beantworteten",
              title="Risk-Coverage", ylim=(0, 1.05))  # fmt: skip
    rc_ax.legend(frameon=False)
    table = {
        "beantwortbar, Treffer": sum(bool(q["ranked"]) for q in questions if q["expect_ids"]),
        "beantwortbar, nichts gefunden": sum(not q["ranked"] for q in questions if q["expect_ids"]),
        "nicht im Laborbuch, Treffer": sum(bool(q["ranked"]) for q in questions if not q["expect_ids"]),
        "nicht im Laborbuch, nichts gefunden": sum(not q["ranked"] for q in questions if not q["expect_ids"]),
    }
    return _png(fig), table


def plot_confusion(result: dict) -> str | None:
    plt = _plt()
    rows = [q for q in result["questions"] if q.get("expect_intent") and not q.get("error")]
    if not rows:
        return None
    labels = sorted({q["expect_intent"] for q in rows} | {q["intent"] for q in rows})
    matrix = np.zeros((len(labels), len(labels)))
    for q in rows:
        matrix[labels.index(q["expect_intent"]), labels.index(q["intent"])] += 1
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="white" if shares[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="vom Planer gewählt", ylabel="erwartet", title=f"Routing (n={len(rows)})")
    return _png(fig)


def plot_latency(result: dict) -> str:
    plt = _plt()
    questions = [q for q in result["questions"] if not q.get("error")]
    steps = [s for s in ("plan", "retrieve", "answer_generate", "answer_total") if any(s in q["latency_ms"] for q in questions)]
    fig, (box_ax, cdf_ax) = plt.subplots(1, 2, figsize=(9, 3))
    data = [[q["latency_ms"][s] / 1000 for q in questions if s in q["latency_ms"]] for s in steps]
    labels = {"plan": "Planung", "retrieve": "Suche", "answer_generate": "Generierung", "answer_total": "Antwort gesamt"}
    box_ax.boxplot(data, tick_labels=[labels[s] for s in steps], showfliers=True)
    box_ax.tick_params(axis="x", labelrotation=15)
    box_ax.set(ylabel="Sekunden", title="Dauer je Schritt")
    total_key = "answer_total" if "answer_total" in steps else "plan"
    totals = sorted(q["latency_ms"][total_key] / 1000 for q in questions if total_key in q["latency_ms"])
    if totals:
        cdf_ax.step(totals, np.arange(1, len(totals) + 1) / len(totals), where="post", color=COLORS[0])
        for pct in (50, 95):
            value = float(np.percentile(totals, pct))
            cdf_ax.axvline(value, color="#999", ls="--", lw=1)
            cdf_ax.text(value, 0.05 + pct / 200, f" p{pct} = {value:.1f} s", fontsize=7)
    cdf_ax.set(xlabel="Sekunden", ylabel="Anteil Fragen", title=f"Verteilung ({'ganze Antwort' if total_key == 'answer_total' else 'Planung'})")
    return _png(fig)


def plot_question_heatmap(result: dict) -> str:
    plt = _plt()
    questions = result["questions"]
    variants = [v["name"] for v in result["variants"]]

    def score(q, variant):
        rank = q["variants"][variant]["rank"]
        if not q["expect_ids"]:
            return 1.0 if not q["variants"][variant]["ranked"] else 0.0  # nothing found is right
        return 1.0 if rank == 1 else 0.6 if rank and rank <= 3 else 0.3 if rank else 0.0

    grid = np.array([[score(q, v) for v in variants] for q in questions])
    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.24 * len(questions)))
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(variants)), variants, rotation=20)
    ax.set_yticks(range(len(questions)), [f"[{q['set'][:5]}] {q['question'][:58]}" for q in questions], fontsize=7)
    ax.set_title("Pro Frage: grün = Platz 1 bzw. richtig nichts gefunden, gelb = Platz 2–3, rot = verfehlt")
    return _png(fig)


# ---------------------------------------------------------------------------
# lab assistant
# ---------------------------------------------------------------------------


def _count(value) -> int:
    """Numbers are stored as a count or as the list of the numbers themselves."""
    if isinstance(value, (list, tuple)):
        return len(value)
    return int(value or 0)


def plot_assistant(data: dict) -> str:
    plt = _plt()
    cases = data.get("cases", [])
    fig, (bar_ax, num_ax, lat_ax) = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={"width_ratios": [3, 2, 1]})
    names = [str(c.get("id")) for c in cases]
    x = np.arange(len(cases))
    metrics = [m for m in ("precision", "supported_precision", "recall", "f1") if any(m in c for c in cases)]
    width = 0.8 / len(metrics)
    labels = {"supported_precision": "precision inkl. Zusätze"}
    for i, metric in enumerate(metrics):
        bar_ax.bar(x + (i - (len(metrics) - 1) / 2) * width, [c.get(metric) or 0 for c in cases], width,
                   color=COLORS[i], label=labels.get(metric, metric))  # fmt: skip
    bar_ax.set_xticks(x, names, rotation=40, ha="right", fontsize=7)
    bar_ax.set(ylim=(0, 1.45), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], title="Messwerte im Entwurf je Fall")
    bar_ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    invented = [_count(c.get("invented_numbers")) for c in cases]
    missing = [_count(c.get("missing_numbers")) for c in cases]
    num_ax.bar(x, invented, color=COLORS[3], label="erfunden")
    num_ax.bar(x, missing, bottom=invented, color=COLORS[5], label="fehlend")
    num_ax.set_xticks(x, names, rotation=40, ha="right", fontsize=7)
    counts = [i + m for i, m in zip(invented, missing)]
    num_ax.set(title="Zahlen ohne Beleg / vergessene Zahlen", ylim=(0, max(3, *counts, 0) + 0.5))
    if not any(counts):
        num_ax.text(0.5, 0.5, "keine in allen Fällen", transform=num_ax.transAxes, ha="center", color="#555")
    else:
        num_ax.legend(frameon=False, fontsize=7)
    lat_ax.boxplot([[c.get("latency_s") or 0 for c in cases]], tick_labels=["Entwurf"])
    lat_ax.set(ylabel="Sekunden", title="Dauer")
    return _png(fig)


# ---------------------------------------------------------------------------
# html
# ---------------------------------------------------------------------------

REFERENCES = [
    ("BEIR (Thakur et al. 2021): nDCG@10, Recall@k als Standard der Retrieval-Evaluation", "https://arxiv.org/abs/2104.08663"),
    ("MTEB (Muennighoff et al. 2022): Benchmark für Embedding-Modelle", "https://arxiv.org/abs/2210.07316"),
    ("RAGAS (Es et al. 2023): Faithfulness, Answer Relevancy, Context Precision/Recall", "https://arxiv.org/abs/2309.15217"),
    ("RAGChecker (Ru et al. 2024): Claim-basierte Diagnose von Retriever und Generator", "https://arxiv.org/abs/2408.08067"),
    ("Selective Classification (Geifman & El-Yaniv 2017): Risk-Coverage-Kurve", "https://arxiv.org/abs/1705.08500"),
    ("Smucker, Allan & Carterette 2007: Signifikanztests in IR, Randomisierungstest empfohlen", "https://doi.org/10.1145/1321440.1321528"),
]


def _table(header: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _figure(png: str | None, caption: str) -> str:
    if not png:
        return ""
    return f'<figure><img src="data:image/png;base64,{png}" alt=""><figcaption>{html.escape(caption)}</figcaption></figure>'


def build_html(result: dict, assistant: dict | None = None) -> str:
    cfg = result["config"]
    metrics_png, metrics_table = plot_metrics_ci(result)
    ablation_png, ablation_table = plot_ablation(result)
    abstention_png, abstention_counts = plot_abstention(result)
    errors = [q for q in result["questions"] if q.get("error")]
    answers = [q["answer"] for q in result["questions"] if q.get("answer")]
    parts = [
        f"<h1>Lab Chatbot und Lab Assistant: Auswertung</h1><p class=meta>{html.escape(result['created'])} · "
        f"Chatmodell <code>{html.escape(cfg['chat_model'])}</code> · Embedding <code>{html.escape(cfg['embed_model'])}</code> · "
        f"{cfg['experiments']} Experimente im Index · Schwelle {cfg['min_dense']}</p>",
        "<p class=note>Kleine Fragensätze: Die Konfidenzintervalle sind breit. Unterschiede, deren Intervall die Null "
        "enthält, sind nicht belegt. Die Werte gelten für dieses Laborbuch und diese Modelle.</p>",
        "<h2>1. Findet die Suche das richtige Experiment?</h2>",
        _figure(plot_hit_at_k(result), "Hit@k: Anteil der Fragen, bei denen das richtige Experiment unter den ersten k ist. "
                "Links je Fragensatz, rechts je Suchvariante."),  # fmt: skip
        _figure(metrics_png, "Hit@1, Hit@3 und MRR (Mittelwert von 1/Rang) mit 95 %-Bootstrap-Intervall über die Fragen."),
        _table(["Fragensatz", "n", "Hit@1", "Hit@3", "MRR", "nDCG@10"], metrics_table),
        "<h2>2. Welcher Baustein bringt wie viel? (Ablation)</h2>",
        _figure(ablation_png, "Gepaarter Unterschied im Reciprocal Rank: Variante minus volles System, je Frage. "
                "Links von 0 = die Variante ist schlechter, der Baustein hilft."),  # fmt: skip
        _table(["Variante", "Δ Reciprocal Rank [95 %-KI]", "p (gepaarter Randomisierungstest)",
                "Treffer bei Fragen ohne Antwort"], ablation_table),  # fmt: skip
        "<p>" + "<br>".join(f"<b>{html.escape(v['name'])}</b>: {html.escape(v['description'])}" for v in result["variants"]) + "</p>",
        "<h2>3. Trennt die Vektorsuche richtig von falsch?</h2>",
        _figure(plot_scores(result), "Ähnlichkeit aller Abschnitte zur Frage. Je weniger sich grün und grau überlappen, "
                "desto besser kann eine Schwelle entscheiden."),  # fmt: skip
        "<h2>4. Erkennt der Chatbot, wenn etwas nicht im Laborbuch steht?</h2>",
        _figure(abstention_png, "Links: ROC über die beste Ähnlichkeit je Frage, beantwortbar gegen nicht im Laborbuch. "
                "Rechts: Fehlerrate, wenn nur die sichersten Fragen beantwortet würden (richtig = Experiment unter den ersten 3)."),  # fmt: skip
        _table(["Fall", "Anzahl"], [[k, v] for k, v in abstention_counts.items()]),
        "<h2>5. Routing des Planers</h2>",
        _figure(plot_confusion(result), "Erwartete gegen gewählte Absicht (nur Fragen mit erwarteter Absicht). "
                "Zahlen = Anzahl Fragen, Farbe = Anteil der Zeile."),  # fmt: skip
        "<h2>6. Antwortzeiten</h2>",
        _figure(plot_latency(result), "Dauer je Schritt und Verteilung der Gesamtzeit (p50/p95). Kalt gestartete Modelle "
                "verlängern die erste Frage."),  # fmt: skip
    ]
    if answers:
        n = len(answers)
        with_llm = [a for a in answers if a["used_llm"]]
        rows = [
            ["Antworten mit Sprachmodell", f"{len(with_llm)}/{n}"],
            ["davon mit mindestens einem Zitat", f"{sum(a['cited'] > 0 for a in with_llm)}/{len(with_llm)}"],
            ["davon mit Zahlen ohne Beleg in den Quellen", f"{sum(a['unsupported_numbers'] for a in with_llm)}/{len(with_llm)}"],
            ["erwarteter Inhalt enthalten", f"{sum(a['contains_ok'] is True for a in answers)}/{sum(a['contains_ok'] is not None for a in answers)}"],
        ]  # fmt: skip
        parts += ["<h2>7. Antworten</h2>", _table(["Kennzahl", "Wert"], rows)]
    parts += ["<h2>8. Einzelne Fragen</h2>", _figure(plot_question_heatmap(result), "Jede Zeile eine Frage, jede Spalte eine Suchvariante.")]
    if assistant:
        summary = assistant.get("summary", {})
        parts += [
            "<h2>9. Lab Assistant: Entwürfe aus Notizen</h2>",
            _figure(plot_assistant(assistant), "Precision/Recall/F1 der erkannten Messwerte (Name, Zahl und Einheit) je Fall, "
                    "Zahlen im Entwurf ohne Beleg in den Notizen, Dauer je Entwurf."),  # fmt: skip
            _table(["Kennzahl", "Wert"], [[k, f"{v:.3f}" if isinstance(v, float) else v] for k, v in summary.items()]),
        ]
    if errors:
        parts.append("<h2>Fehler</h2>" + _table(["Frage", "Fehler"], [[q["question"], q["error"]] for q in errors]))
    parts.append("<h2>Grundlagen</h2><ul>" + "".join(
        f'<li><a href="{url}">{html.escape(text)}</a></li>' for text, url in REFERENCES) + "</ul>")  # fmt: skip
    style = (
        "body{font-family:-apple-system,Segoe UI,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}"
        "h1{font-size:22px}h2{font-size:17px;margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}"
        "img{max-width:100%}figure{margin:12px 0}figcaption{font-size:12px;color:#555}"
        "table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ddd;padding:4px 8px;text-align:left}"
        ".meta{color:#555;font-size:12px}.note{background:#f5f5f0;padding:8px 12px;font-size:12px}"
    )
    return f"<!doctype html><html lang=de><meta charset=utf-8><title>Auswertung Lab Chatbot</title><style>{style}</style><body>{''.join(parts)}</body></html>"


def write_report(result: dict, out_dir: str | Path, assistant: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.html"
    path.write_text(build_html(result, assistant), encoding="utf-8")
    return path
