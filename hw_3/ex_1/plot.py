import matplotlib.pyplot as plt
import pandas as pd


df = pd.DataFrame([['131070', 331, 172, 525], ['262140', 590, 248, 945], ['524280', 1297, 227, 1524],
				['1048560', 2504, 299, 2849]],
				columns=['Input Length', 'h2d', 'kernel', 'd2h'])


df.plot(x='Input Length',ylabel='Latency (µs)',xlabel='Input Length', kind='bar', stacked=True,
		title='Task latency of operations')
plt.show()
