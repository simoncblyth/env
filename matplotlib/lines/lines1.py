"""
lines1.py
==========

Draws two line segments joining the below three points::

                       +
                       (100,300)

                
         +  
         (10,30)
   +
  (1,3)

"""
import numpy as np
import matplotlib.pyplot as plt

line = np.array([[1,2,3],[10,20,30],[100,200,300]]) 
H, V = 0, 2 

plt.plot( line[:,H], line[:,V], 'ro-')

plt.show()
