#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

x = [u'INFO', u'CUISINE', u'TYPE_OF_PLACE', u'DRINK', u'PLACE', u'MEAL_TIME', u'DISH', u'NEIGHBOURHOOD']
y = [160, 167, 137, 18, 120, 36, 155, 130]

fig, ax = plt.subplots()    

width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups

ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)

plt.title('title')
plt.xlabel('x')
plt.ylabel('y')      

plt.show()
#plt.savefig(os.path.join('test.png'), dpi=300, format='png', bbox_inches='tight') 
