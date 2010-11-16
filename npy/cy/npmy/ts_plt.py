"""

  ipython plt.py -pylab 

  
"""

import matplotlib
matplotlib.use('TkAgg')       # Agg      non-interactive/web server usage
import matplotlib.pyplot as pyplot
fig = pyplot.figure()
ax = fig.add_subplot(111)

from test_npmy import Fetch
npz = Fetch.scan("DcsPmtHv")
ts = npz['ts']
    

if 0:
    # when multiple methods in the same structure .. reshape it for convenience
    ts.shape = (2, ts.shape[0]/2)
    print ts[0]['time']
    print ts[1]['time']
    ax.plot( ts[0]['limit'], ts[0]['time'] , "ro" , ts[1]['limit'], ts[1]['time'] , "bo" ) 
else:
    ax.plot( ts['limit'], ts['time'] , "ro"  ) 


fig.show()


