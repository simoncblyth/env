"""
  Reading an apache access_log into a numpy array 
     * the datetime format in the log [17/Jan/2011:11:48:20 +0800] does not get correctly converted
       to an M8[s]         

In [6]: tt = np.ones( 10 ,  dtype="M8[ms]" )
In [7]: tt[0] = "17/Jan/2011:11:48:10 +0800"

In [24]: t.strftime("%d/%b/%Y:%H:%M:%S")
Out[24]: '17/Jan/2011:13:32:27'

"""
import os
import numpy as np
from StringIO import StringIO

#s = os.path.expanduser("~/access_log")
s = StringIO(r"""207.46.13.131 - - [17/Jan/2011:11:48:10 +0800] "GET /tracs/env/ticket/228 HTTP/1.1" 200 14270
77.88.27.27 - - [17/Jan/2011:11:48:20 +0800] "GET /tracs/env/export/3255/trunk/vdbi/vdbi/tw/jquery/__init__.py HTTP/1.1" 200 -
77.88.27.27 - - [17/Jan/2011:11:48:29 +0800] "GET /tracs/env/changeset/43 HTTP/1.1" 200 9764
77.88.27.27 - - [17/Jan/2011:11:48:41 +0800] "GET /tracs/env/changeset/1908/trunk/nuwa HTTP/1.1" 200 15076
77.88.27.27 - - [17/Jan/2011:11:49:14 +0800] "GET /tracs/env/wiki/ApacheBenchmarks HTTP/1.1" 200 51613
77.88.27.27 - - [17/Jan/2011:11:49:26 +0800] "GET /tracs/tracdev/changeset/55/trac2mediawiki/trunk/0.11/plugins/trac2mediawiki/__init__.py HTTP/1.1" 200 10794
77.88.27.27 - - [17/Jan/2011:11:49:36 +0800] "GET /repos/env/trunk/dyw/root_use.bash HTTP/1.1" 200 1821
77.88.27.27 - - [17/Jan/2011:11:49:44 +0800] "GET /tracs/env/changeset/2422/trunk/mysql HTTP/1.1" 200 10162
77.88.27.27 - - [17/Jan/2011:11:49:58 +0800] "GET /tracs/env/changeset/2823/trunk/#hello HTTP/1.1" 200 13307
77.88.27.27 - - [17/Jan/2011:11:50:09 +0800] "GET /tracs/env/changeset/2082/trunk/thho HTTP/1.1" 200 6551
""")

t = np.dtype([ ('ip','S15'), ('id', 'S10') , ('user', 'S10'), ('time','S26'), ('req','S100'), ('resp','i4'), ('size','i4') ] )   
r = r'(\d+\.\d+\.\d+\.\d+) (.?) (.?) \[(.*)\] "(.*)" (\d+) (\d+)'

aa = np.fromregex( s, r, t )
a = aa.view(np.recarray)



