import matplotlib.pyplot as plt
import numpy as np

rawData = ""

with open('data.txt') as f:
    rawData = f.readlines()

dataStrArray = rawData[0].split(",")[:-1]
x = [int(i) for i in dataStrArray]
#print(x)

bins = [i for i in range(4096)]
plt.bar(bins, x)
plt.xlabel("Bins")
plt.ylabel("Occurance")
plt.title("Input length: 512000, grid size: 1000, block size: 512")
plt.show()