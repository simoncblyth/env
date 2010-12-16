
import numpy as np
from env.mysql_numpy import DB
from pylab import *

LEVEL = 300.0


db = DB()
a = db("select SEQNO, VOLTAGE from DcsPmtHv where ladder=1 and col=1 and ring=1 and voltage is not NULL")

arrlen = 0

for element in a["VOLTAGE"]:
    if element > 0:
        arrlen = arrlen + 1

arrseq = np.zeros(arrlen)
arrvlt = np.zeros(arrlen)

arroldcounter = 0
arrnewcounter = 0
for element in a["VOLTAGE"]:
    if element > LEVEL:
        arrseq[arrnewcounter] = a["SEQNO"][arroldcounter]
        arrvlt[arrnewcounter] = element
        arrnewcounter = arrnewcounter + 1
    arroldcounter = arroldcounter + 1


plot(arrseq, arrvlt, 'o')

show()

