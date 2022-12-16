import matplotlib.pyplot as plt
import pandas as pd

# float type 
df = pd.DataFrame([['512', 692, 1019, 897], ['768', 1447, 2091, 1690], ['1024', 2434, 8961, 3076]],
				columns=['Input Length', 'h2d', 'kernel', 'd2h'])

# double type
#df = pd.DataFrame([['512', 1181, 2384, 1520], ['768', 2857, 14356, 3415], ['1024', 4657, 21843, 3915]],
#				columns=['Input Length', 'h2d', 'kernel', 'd2h'])


df.plot(x='Input Length',ylabel='Latency (µs)',xlabel='Square Matrix Dimention', kind='bar', stacked=True,
		title='Task latency of operations')
plt.show()
