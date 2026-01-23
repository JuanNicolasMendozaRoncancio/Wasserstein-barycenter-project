import matplotlib.pyplot as plt
import numpy as np

l = [0,2,4,12,14,16]

def f(x):
    return sum([np.maximum(n-x,0) for n in l])


l_sorted = sorted(l, reverse=True)
l_cum_sum = np.cumsum(l_sorted)
frac = [l_sorted[i] > (sum-7.5)/(i+1) for i, sum in enumerate(l_cum_sum)]

k = max([i for i, val in enumerate(frac) if val])

print("k:", k)

x = np.linspace(0,16,100)
y = [f(xi) for xi in x]


plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('\phi(x)')
plt.title('Function φ(x) = Σ max(li - x, 0)')

plt.vlines(l, ymin=0, ymax=max(y), colors='r', linestyles='dashed', label='li values')
plt.hlines(7.5, xmin=0, xmax=16, colors='b', linestyles='dashed', label='u = 7.5')

plt.grid()
plt.show()

print("Sorted l:", l_sorted)
print("Cumulative sums:", l_cum_sum)
print("Fractions:", frac)
