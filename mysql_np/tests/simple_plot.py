"""
   http://matplotlib.sourceforge.net/faq/installing_faq.html

(npy)[blyth@cms01 tests]$ python simple_plot.py --verbose-helpful
$HOME=/home/blyth
CONFIGDIR=/home/blyth/.matplotlib
matplotlib data path /data/env/local/env/v/npy/src/matplotlib/lib/matplotlib/mpl-data
loaded rc file /data/env/local/env/v/npy/src/matplotlib/lib/matplotlib/mpl-data/matplotlibrc
matplotlib version 1.0.0
verbose.level helpful
interactive is False
units is False
platform is linux2
Using fontManager instance from /home/blyth/.matplotlib/fontList.cache
backend TkAgg version 8.4
findfont: Matching :family=sans-serif:style=normal:variant=normal:weight=normal:stretch=normal:size=medium to Bitstream Vera Sans (/data/env/local/env/matplotlib/matplotlib/lib/matplotlib/mpl-data/fonts/ttf/Vera.ttf) with score of 0.000000


"""
from pylab import *
plot([1,2,3])
show()

