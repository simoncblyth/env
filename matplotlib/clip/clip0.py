"""
http://stackoverflow.com/questions/10192011/clipping-in-matplotlib-why-doesnt-this-work

The clipping chops off the top half of the circle

"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

plt.ion()


def make_rect(bl, tr, **kwa):
    """
    :param bl: bottom left coordinates tuple
    :param tr: top right coordinates tuple
    """
    return Rectangle( bl, tr[0]-bl[0], tr[1]-bl[1], **kwa)

fig = plt.figure()
ax = fig.add_subplot(121, aspect='equal')

rect = make_rect((0.5,0.5), (3,3), facecolor="none", edgecolor="none")

circle = Circle((0,0),2, fc="none", ec="b")
circle.mybbox = rect 

ax.add_artist(rect)
ax.add_artist(circle)

# clipping only works after both have been added
#circle.set_clip_path(rect)  

# workaround that inconvenience
for a in ax.findobj(lambda a:hasattr(a, 'mybbox')):
    print("set_clip_path on %r " % a )
    a.set_clip_path(a.mybbox)


plt.axis((-4,4,-4,4))
plt.show()
