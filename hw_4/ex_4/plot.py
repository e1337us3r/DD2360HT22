import matplotlib.pyplot as plt
import subprocess

y = [] # relative error 
x = [] # step count

for i in range(100, 10001, 100):
    result = subprocess.run([f"./run.o 128 {i}"], shell=True, capture_output=True, text=True)
    x.append(i)
    y.append(float(result.stdout.split(" ")[-1]))

# Make the plot
plt.plot(x,y)

# Adding Xticks
plt.xlabel('Nsteps', fontweight ='bold', fontsize = 15)
plt.ylabel('Relative error', fontweight ='bold', fontsize = 15)
plt.title("Relationship between nsteps and approximation error")

plt.show()
