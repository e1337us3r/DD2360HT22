import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

# set height of bar
DEFAULT = [ 50, 244, 480, 932]
STREAM = [ 27, 139, 279, 564]

# Set position of bar on X axis
br1 = np.arange(len(DEFAULT))
br2 = [x + barWidth for x in br1]

# Make the plot
plt.bar(br1, DEFAULT, color ='b', width = barWidth,
		edgecolor ='grey', label ='Default')
plt.bar(br2, STREAM, color ='g', width = barWidth,
		edgecolor ='grey', label ='Stream')

# Adding Xticks
plt.xlabel('Input length', fontweight ='bold', fontsize = 15)
plt.ylabel('Total runtime (ms)', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(DEFAULT))],
		[ '10M', '50M', '100M', '200M'])
plt.title("Vector addition kernel, Default vs with 4 Streams")

plt.legend()
plt.show()
