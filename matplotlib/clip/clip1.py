"""
http://stackoverflow.com/questions/10192011/clipping-in-matplotlib-why-doesnt-this-work
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

rect = Rectangle((-2,-2),4,2, facecolor="none", edgecolor="none")
circle = Circle((0,0),1)

plt.axes().add_artist(rect)
plt.axes().add_artist(circle)

circle.set_clip_path(rect)

plt.axis('equal')
plt.axis((-2,2,-2,2))
plt.show()
