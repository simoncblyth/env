#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec


f = plt.figure()

#gs = gridspec.GridSpec(1, 2,width_ratios=[2,1])
gs = gridspec.GridSpec(2, 1, height_ratios=[2,1])

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

plt.show()

