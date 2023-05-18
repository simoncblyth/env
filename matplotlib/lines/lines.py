

import numpy as np
import matplotlib.pyplot as plt

x, y = np.random.random(size=(2,10))

print("x",x)
print("y",y)


for i in range(0, len(x), 2):

    print("i:%d " % i )
    print("x[i:i+2]:%s" % str(x[i:i+2]))
    print("y[i:i+2]:%s" % str(y[i:i+2]))

    plt.plot(x[i:i+2], y[i:i+2], 'ro-')
pass

plt.show()
