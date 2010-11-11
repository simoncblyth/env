
from pylab import *

x = arange(10)
y = x

# Plot junk and then a filled region
plot(x, y)

# Make a blue box that is somewhat see-through
# and has a red border. 
# WARNING: alpha doesn't work in postscript output.... 
fill([3,4,4,3], [2,2,4,4], 'b', alpha=0.2, edgecolor='r')


#
#
#     x1,x2,x3,x4  y1,y2,y3,y4
#  general polygon specification  (x1,y1),(x2,y2),...,(x4,y4)
#
#  for an axis aligned rectangle ...  specified by left bottom and width,height 
#     [x0,x0+w,x0+w,x0 ] [y0,y0,y0+h,y0+h] 
#
#  or specifying the corners 
#     [x0,x1,x1,x0]    [y0,y0,y1,y1]
#


