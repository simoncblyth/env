#!/usr/bin/env python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
plt.ion()


np.random.seed(104)



cities = ['New York', 'Boston', 'San Francisco', 'Seattle', 'Los Angeles', 'Chicago']

car_sales = pd.DataFrame({
 'sales':np.random.randint(1e3,1e4,len(cities)),
 'goal':np.random.randint(4e3,8e3,len(cities)),
 'sales_last_year':np.random.randint(1e3,1e4,len(cities)),
}, index=cities)

car_sales_sorted = car_sales.sort_values('sales',ascending=False)



if 0:

    ax = car_sales.sales.plot(kind='bar')

    ax = car_sales_sorted.sales.plot(kind='bar',facecolor='#AA0000')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.patch.set_facecolor('#FFFFFF')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(1)

    plt.show()




if 0:

    ax = car_sales_sorted.sales.plot(
     kind='bar',
     color=['#AA0000' if row.sales > row.goal else '#000088' for name,row in car_sales_sorted.iterrows()],
     )

    percent_of_goal = ["{}%".format(int(100.*row.sales/row.goal)) for name,row in car_sales_sorted.iterrows()]
    for i,child in enumerate(ax.get_children()[:car_sales_sorted.index.size]):
        ax.text(i,child.get_bbox().y1+200,percent_of_goal[i], horizontalalignment ='center')

    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.patch.set_facecolor('#FFFFFF')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(1)
    plt.show()

 

if 0:

    ax = car_sales_sorted[['sales','sales_last_year']].plot(
     kind='bar',
     )

    percent_of_goal = ["{}%".format(int(100.*row.sales/row.goal)) for name,row in car_sales_sorted.iterrows()]
    pairs = len(cities)
    make_pairs = zip(*[ax.get_children()[:pairs],ax.get_children()[pairs:pairs*2]])
    for i,(left, right) in enumerate(make_pairs):
        ax.text(i,max(left.get_bbox().y1,right.get_bbox().y1)+200,percent_of_goal[i], horizontalalignment ='center')

    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.patch.set_facecolor('#FFFFFF')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(1)

    plt.show()
     


if 1: 

    

    ax = car_sales_sorted.sales.plot(
     kind='barh',
     color=['#AA0000' if row.sales > row.goal else '#000088' for name,row in car_sales_sorted.iterrows()],
     )

    percent_of_goal = ["{}%".format(int(100.*row.sales/row.goal)) for name,row in car_sales_sorted.iterrows()]
    for i,child in enumerate(ax.get_children()[:car_sales_sorted.index.size]):
        ax.text(child.get_bbox().x1+200,i,percent_of_goal[i], verticalalignment ='center')

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.patch.set_facecolor('#FFFFFF')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(1)
    ax.yaxis.set_ticklabels([
     'Seattle: City of Fish',
     'New York: City of Taxis',
     'Los Angeles: City of Stars and Smog',
     'Boston: City of Sox',
     'Chicago: City of Bean',
     'San Francisco: City of Rent'
     ], fontstyle='italic')#={'weight':'bold'}

    make_axes_area_auto_adjustable(ax)
    plt.show()


