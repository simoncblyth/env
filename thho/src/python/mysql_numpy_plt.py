#
# use
# % ipython -pylab
# to set up the enviroment equal to
# import numpy as np
# import matplotlib.pyplot as plt
#
#
import numpy as np
import matplotlib.pyplot as plt
import MySQLdb
db = MySQLdb.connect(read_default_file="/home/thho/.my.cnf")
c = db.cursor()
c.execute("""select PMTSPEHIGH from CalibPmtSpec""")
tmp = c.fetchmany(10000)
a = np.array(tmp)
#plt.hist(a, 50)
#plt.show()
