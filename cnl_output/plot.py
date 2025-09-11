import pandas as pd
import matplotlib.pyplot as plt

# Load data
llama = pd.read_csv("./llama_coverage/llama_coverage.txt", delim_whitespace=True, skiprows=2)
mistral = pd.read_csv("./mistral_coverage/mistral_coverage.txt", delim_whitespace=True, skiprows=2)

# Clean column names
llama.columns = ["match_type", "n_words", "matched", "total", "coverage_pct"]
mistral.columns = ["match_type", "n_words", "matched", "total", "coverage_pct"]

# Filter relevant rows (forward and backward)
llama_fwd = llama[llama["match_type"] == "forward_first_n"]
llama_bwd = llama[llama["match_type"] == "backward_last_n"]
mistral_fwd = mistral[mistral["match_type"] == "forward_first_n"]
mistral_bwd = mistral[mistral["match_type"] == "backward_last_n"]

# Plot
plt.figure(figsize=(7, 5))

# Forward plots (solid lines)
plt.plot(llama_fwd["n_words"], llama_fwd["coverage_pct"], 'o-', color='red', label="Llama - Forward")
plt.plot(mistral_fwd["n_words"], mistral_fwd["coverage_pct"], 'o-', color='blue', label="Mistral - Forward")

# Backward plots (dashed lines)
plt.plot(llama_bwd["n_words"], llama_bwd["coverage_pct"], 's--', color='red', label="Llama - Backward")
plt.plot(mistral_bwd["n_words"], mistral_bwd["coverage_pct"], 's--', color='blue', label="Mistral - Backward")

# Labels and title
plt.xlabel("n (words)")
plt.ylabel("Coverage (%)")
plt.title("Coverage vs. n (words)")

# Custom legend (shows dashes properly)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', marker='o', linestyle='-', label="Llama - Forward"),
    Line2D([0], [0], color='blue', marker='o', linestyle='-', label="Mistral - Forward"),
    Line2D([0], [0], color='red', marker='s', linestyle='--', label="Llama - Backward"),
    Line2D([0], [0], color='blue', marker='s', linestyle='--', label="Mistral - Backward")
]
plt.legend(
    handles=legend_elements,
    loc="best",
    fontsize=14,      # legend text size
    markerscale=1.5,  # make legend markers bigger
    handlelength=3,   # extend line dashes inside legend
    handletextpad=0.8 # spacing between line and text
)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save and show
out_path = "coverage.png"
plt.savefig(out_path, dpi=300)
plt.close()

out_path
