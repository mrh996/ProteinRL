import pandas as pd
import matplotlib.pyplot as plt

# Example DataFrame (Assuming the data is stored in a CSV or directly in a DataFrame)
data = {
    "Protein": ["1qjp", "1c5g", "1cah", "2imm", "2abd", "1rgg", "1arr", "2lzm", "1rtp", "1div", "1ag2", "1ank"],
    "BO-ENN": [-0.137555, -0.719112, -0.502136, -10, -1.479002, -2.572706, -0.920365, -0.624346, -0.021404, -0.638530, -0.075541, -0.531443],
    "BO": [0.326877, -0.257632, 0.289356, 0.150912, -0.232414, -0.016236, -0.304492, -0.427544, 0.062406, -0.332000, -0.081967, -0.314967],
    "Random": [-2.158175, -1.670554, -2.694460, -1.642179, -2.009476, -2.292602, -2.428045, -2.773323, -3.212715, -2.354292, -2.536845, -2.713916],
}

df = pd.DataFrame(data)

# Calculate mean and std for each column
means = df[["BO-ENN", "BO", "Random"]].mean()
stds = df[["BO-ENN", "BO", "Random"]].std()

# Plot the mean and standard deviation
plt.figure(figsize=(10, 6))

# Plot mean values with standard deviation as error bars
plt.bar(means.index, means, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])

# Set titles and labels
plt.title('Mean and Standard Deviation of Columns')
plt.xlabel('Columns')
plt.ylabel('Values')

# Save the figure to a file (e.g., PNG, PDF)
plt.savefig('/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/mean_std_chart.png', dpi=300, bbox_inches='tight')

# Optionally, you can show the plot as well
plt.show()

