"""
Generate academic evaluation tables and charts from results.
Requires: pip install matplotlib pandas tabulate
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "results"
FIGURES_DIR = "../figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi':     150,
})


def load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  Missing: {filename}")
        return None
    with open(path) as f:
        return json.load(f)


def plot_intent_classification(data):
    if not data:
        return

    classes    = list(data["per_class"].keys())
    precision  = [data["per_class"][c]["precision"] for c in classes]
    recall     = [data["per_class"][c]["recall"]    for c in classes]

    x     = range(len(classes))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # bar chart
    ax = axes[0]
    bars1 = ax.bar(
        [i - width/2 for i in x], precision,
        width, label='Precision', color='black'
    )
    bars2 = ax.bar(
        [i + width/2 for i in x], recall,
        width, label='Recall', color='gray'
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Intent Classification: Precision and Recall')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    # summary metrics
    ax2 = axes[1]
    ax2.axis('off')

    metrics = [
        ["Metric",              "Value"],
        ["Total test cases",    str(data["total"])],
        ["Intent accuracy",     f"{data['intent_accuracy']*100:.1f}%"],
        ["Tool accuracy",       f"{data['tool_accuracy']*100:.1f}%"],
        ["Macro F1-score",      f"{data['macro_f1']:.3f}"],
        ["Misclassified",       str(data["total"] - data["correct_intent"])],
    ]

    table = ax2.table(
        cellText  = metrics[1:],
        colLabels = metrics[0],
        loc       = 'center',
        cellLoc   = 'center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    ax2.set_title('Summary Metrics')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "intent_classification.png")
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def plot_latency(data):
    if not data:
        return

    categories = list(data.keys())
    avg_total  = [data[c]["avg_total_ms"]       for c in categories]
    avg_first  = [data[c]["avg_first_chunk_ms"]  for c in categories]

    labels = [c.replace("_", " ").title() for c in categories]
    x      = range(len(categories))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    b1 = ax.bar(
        [i - width/2 for i in x], avg_total,
        width, label='Total latency', color='black'
    )
    b2 = ax.bar(
        [i + width/2 for i in x], avg_first,
        width, label='Time to first chunk', color='gray'
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel('Milliseconds (ms)')
    ax.set_title('Response Latency by Query Type')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 20,
                f'{h:.0f}', ha='center', va='bottom', fontsize=8)

    # comparison table
    ax2 = axes[1]
    ax2.axis('off')

    rows  = [["Category", "Avg (ms)", "First chunk (ms)", "Min", "Max"]]
    for cat in categories:
        d = data[cat]
        rows.append([
            cat.replace("_", " ").title(),
            str(d["avg_total_ms"]),
            str(d["avg_first_chunk_ms"]),
            str(d["min_ms"]),
            str(d["max_ms"])
        ])

    table = ax2.table(
        cellText  = rows[1:],
        colLabels = rows[0],
        loc       = 'center',
        cellLoc   = 'center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.4, 1.5)
    ax2.set_title('Latency Breakdown')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "latency.png")
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def plot_human_eval(data):
    if not data:
        return

    by_cat = data["by_category"]
    cats   = list(by_cat.keys())
    scores = [by_cat[c] for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # horizontal bar chart
    ax = axes[0]
    bars = ax.barh(
        range(len(cats)), scores,
        color=['black' if s >= 4 else 'gray' for s in scores]
    )
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel('Score (1–5)')
    ax.set_title('Human Evaluation Scores by Category')
    ax.axvline(x=data["overall_avg"], color='black',
               linestyle='--', linewidth=1, alpha=0.6,
               label=f'Overall avg: {data["overall_avg"]:.2f}')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.05, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=9)

    # distribution
    ax2  = axes[1]
    resp = data.get("responses", [])
    if resp:
        all_scores = [r["score"] for r in resp]
        dist       = [all_scores.count(i) for i in range(1, 6)]
        ax2.bar(range(1, 6), dist, color='black', alpha=0.8)
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Score Distribution (Overall avg: {data["overall_avg"]:.2f}/5)')
        ax2.set_xticks(range(1, 6))
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "human_eval.png")
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def plot_wake_word(data):
    if not data:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    metrics = ['FAR', 'FRR', 'Accuracy']
    values  = [
        data['far']      * 100,
        data['frr']      * 100,
        data['accuracy'] * 100
    ]
    colors  = ['gray', 'gray', 'black']

    bars = ax.bar(metrics, values, color=colors)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Wake Word Detection — Threshold {data["threshold"]}')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.1f}%', ha='center', va='bottom')

    cm_labels = [
        f'TP={data["tp"]}',
        f'FP={data["fp"]}',
        f'TN={data["tn"]}',
        f'FN={data["fn"]}'
    ]
    ax.text(0.98, 0.95, '\n'.join(cm_labels),
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "wake_word.png")
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def print_latex_tables(
    intent_data, latency_data,
    human_data,  wake_data
):
    print("\n" + "═"*60)
    print("LATEX TABLES — paste into your report")
    print("═"*60)

    if intent_data:
        print("""
\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\
\\hline""")
        for cls, r in intent_data["per_class"].items():
            p  = r["precision"]
            rc = r["recall"]
            f1 = 2*p*rc/(p+rc) if (p+rc) > 0 else 0
            print(f"{cls.capitalize()} & {p:.3f} & {rc:.3f} & {f1:.3f} \\\\")
        print(f"\\hline")
        print(f"\\textbf{{Overall}} & \\multicolumn{{3}}{{c|}}{{Accuracy: {intent_data['intent_accuracy']*100:.1f}\\%  —  Macro F1: {intent_data['macro_f1']:.3f}}} \\\\")
        print("""\\hline
\\end{tabular}
\\caption{Intent Classification Results}
\\label{tab:intent}
\\end{table}""")

    if latency_data:
        print("""
\\begin{table}[h!]
\\centering
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Query Type} & \\textbf{Avg (ms)} & \\textbf{1st Chunk (ms)} & \\textbf{Min} & \\textbf{Max} \\\\
\\hline""")
        for cat, d in latency_data.items():
            label = cat.replace("_", " ").title()
            print(
                f"{label} & {d['avg_total_ms']} & "
                f"{d['avg_first_chunk_ms']} & "
                f"{d['min_ms']} & {d['max_ms']} \\\\"
            )
        print("""\\hline
\\end{tabular}
\\caption{Response Latency Benchmark}
\\label{tab:latency}
\\end{table}""")

    if human_data:
        print(f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Category}} & \\textbf{{Avg Score (1--5)}} \\\\
\\hline""")
        for cat, score in human_data["by_category"].items():
            print(f"{cat} & {score:.2f} \\\\")
        print(f"\\hline")
        print(f"\\textbf{{Overall}} & \\textbf{{{human_data['overall_avg']:.2f}}} \\\\")
        print("""\\hline
\\end{tabular}
\\caption{Human Evaluation Scores}
\\label{tab:human}
\\end{table}""")


if __name__ == "__main__":
    print("\nGenerating evaluation figures and LaTeX tables...")

    intent_data  = load("intent_classification_results.json")
    latency_data = load("latency_results.json")
    human_data   = load("human_eval_results.json")
    wake_data    = load("wake_word_results.json")

    plot_intent_classification(intent_data)
    plot_latency(latency_data)
    plot_human_eval(human_data)
    plot_wake_word(wake_data)
    print_latex_tables(
        intent_data, latency_data,
        human_data,  wake_data
    )
    print("\nDone.")