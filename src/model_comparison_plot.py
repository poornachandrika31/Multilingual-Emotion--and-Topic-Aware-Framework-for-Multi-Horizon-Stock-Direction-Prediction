import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.read_csv("results/model_results_all_horizons.csv")

plt.figure(figsize=(10,6))

sns.barplot(
    data=results,
    x="Experiment",
    y="Accuracy",
    hue="Model"
)

plt.title("Model Performance Comparison")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("results/model_comparison_accuracy.png")

plt.show()