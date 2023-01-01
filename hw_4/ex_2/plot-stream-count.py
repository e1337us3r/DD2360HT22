import matplotlib.pyplot as plt

y = [ 558, 540, 538, 585]
x = [ '4', '10', '20', '100']


# Make the plot
plt.plot(x,y)

# Adding Xticks
plt.xlabel('Segment size', fontweight ='bold', fontsize = 15)
plt.ylabel('Total runtime (ms)', fontweight ='bold', fontsize = 15)
plt.title("Segment size comparison, 200M input size")

plt.show()
