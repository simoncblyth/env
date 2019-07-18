#!/usr/bin/env python
"""
https://matplotlib.org/3.1.1/gallery/axes_grid1/make_room_for_ylabel_using_axesgrid.html#sphx-glr-gallery-axes-grid1-make-room-for-ylabel-using-axesgrid-py

"""

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

import matplotlib.pyplot as plt
plt.ion()

if __name__ == "__main__":


    def ex1():
        plt.figure(1)
        ax = plt.axes([0, 0, 1, 1])
        #ax = plt.subplot(111)

        ax.set_yticks([0.5])
        ax.set_yticklabels(["very long label very long label"])

        make_axes_area_auto_adjustable(ax)


    def ex1b():

        x = [u'INFO', u'CUISINE', u'TYPE_OF_PLACE', u'DRINK', u'PLACE', u'MEAL_TIME', u'DISH', u'NEIGHBOURHOOD']
        y = [160, 167, 137, 18, 120, 36, 155, 130]

        fig, ax = plt.subplots()    

        width = 0.75 # the width of the bars 
        ind = np.arange(len(y))  # the x locations for the groups

        ax.barh(ind, y, width, color="blue")
        ax.set_yticks(ind+width/2)
        ax.set_yticklabels(x, minor=False)

        make_axes_area_auto_adjustable(ax)


    def ex2():

        plt.figure(2)
        ax1 = plt.axes([0, 0, 1, 0.5])
        ax2 = plt.axes([0, 0.5, 1, 0.5])

        ax1.set_yticks([0.5])
        ax1.set_yticklabels(["very long label very long label"])
        ax1.set_ylabel("Y label")

        ax2.set_title("Title")

        make_axes_area_auto_adjustable(ax1, pad=0.1, use_axes=[ax1, ax2])
        make_axes_area_auto_adjustable(ax2, pad=0.1, use_axes=[ax1, ax2])

    def ex3():

        fig = plt.figure(3)
        ax1 = plt.axes([0, 0, 1, 1])
        divider = make_axes_locatable(ax1)

        ax2 = divider.new_horizontal("100%", pad=0.3, sharey=ax1)
        ax2.tick_params(labelleft=False)
        fig.add_axes(ax2)

        divider.add_auto_adjustable_area(use_axes=[ax1], pad=0.1,
                                         adjust_dirs=["left"])
        divider.add_auto_adjustable_area(use_axes=[ax2], pad=0.1,
                                         adjust_dirs=["right"])
        divider.add_auto_adjustable_area(use_axes=[ax1, ax2], pad=0.1,
                                         adjust_dirs=["top", "bottom"])

        ax1.set_yticks([0.5])
        ax1.set_yticklabels(["very long label very long label"])

        ax2.set_title("Title")
        ax2.set_xlabel("X - Label")

    #ex1()
    ex1b()
    #ex2()
    #ex3()

    plt.show()
