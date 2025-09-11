import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["Dense (S≤3)", "Unnatural (N≤3)", "Ambiguous (P≤3)", "High-quality (P,N,S≥4)"]
llama_no_gen = [12, 12, 7, 8]
mistral_no_gen = [15, 16, 5, 7]
llama_with_gen = [7, 1, 0, 17]
mistral_with_gen = [5, 2, 1, 20]

x = np.arange(len(categories))  # label locations
width = 0.35  # width of bars

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Define y-axis ticks as multiples of 4 up to max value
y_max = max(max(llama_no_gen + mistral_no_gen), max(llama_with_gen + mistral_with_gen)) + 2
y_ticks = np.arange(0, y_max + 1, 4)

# Without generated RQs
axes[0].bar(x - width/2, llama_no_gen, width, label="Llama 3.2", color="red")
axes[0].bar(x + width/2, mistral_no_gen, width, label="Mistral", color="blue")
axes[0].set_title("Templates without generated RQs")
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, rotation=20, ha="right")
axes[0].set_ylabel("Count")
axes[0].set_yticks(y_ticks)
axes[0].legend(fontsize=10)

# With generated RQs
axes[1].bar(x - width/2, llama_with_gen, width, label="Llama 3.2", color="red")
axes[1].bar(x + width/2, mistral_with_gen, width, label="Mistral", color="blue")
axes[1].set_title("Templates with generated RQs")
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, rotation=20, ha="right")
axes[1].set_ylabel("Count")
axes[1].set_yticks(y_ticks)
axes[1].legend(fontsize=10)

plt.suptitle("PENS Evaluation Results (25 templates per model)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()