import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.ion()

fig, ax = plt.subplots(1,1)
plt.show()

data = np.random.randn(10)
ax.plot(data)


fig.canvas.draw()


