import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

# create the figure and the axis in one shot
fig, axs = plt.subplots(1,2,figsize=(6,6))

# hmm seems that cannot plot the same patch on multiple ax
# attempting to do so yields a blank

for ax in axs:
    art = mpatches.Circle([0,0], radius = 1, color = 'r')
    ax.add_patch(art)

for ax in axs:
    art = mpatches.Circle([0,0], radius = 0.1, color = 'b')
    ax.add_patch(art)

#print ax.patches

#set the limit of the axes to -3,3 both on x and y

axs[0].set_xlim(-1,1)
axs[0].set_ylim(-1,1)

axs[1].set_xlim(-2,2)
axs[1].set_ylim(-2,2)




plt.show()
