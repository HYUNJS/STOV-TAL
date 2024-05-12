import matplotlib.pyplot as plt

a = [0.1, 0.2, 0.3, 0.4, 0.5]
b = [21.1, 23.0, 23.4, 23.5, 23.4]
c = [8.2, 9.1, 9.4, 9.1, 9.1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
# Plotting on the first subplot
line1, = ax1.plot(a, b, 'r-', marker='o', label='K400')
ax1.set_xlabel('Threshold Value')
ax1.set_ylabel('mAP')
ax1.set_xticks(a)  # Set x-ticks to the values in 'a'
# Plotting on the second subplot
line2, = ax2.plot(a, c, 'b-', marker='s', label='non-K400')
ax2.set_xlabel('Threshold Value')
ax2.set_ylabel('mAP')
ax2.set_xticks(a)  # Set x-ticks to the values in 'a'
# Creating a combined legend for the whole figure
plt.figlegend([line1, line2], ['K400', 'non-K400'], loc='upper center', ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend
plt.savefig('ablation_graph.png', dpi=300)
plt.show()