import pandas as pd
import matplotlib.pyplot as plt

# Load your summary results
df = pd.read_csv("fairness_summary.csv")

# Plot IDI vs Budget
plt.figure()
plt.plot(df["budget"], df["mean"], marker='o')

plt.xlabel("Budget")
plt.ylabel("IDI Ratio")
plt.title("IDI Ratio vs Budget")

# Save the graph
plt.savefig("idi_vs_budget.png")

plt.show()
