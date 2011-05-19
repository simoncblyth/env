"""
Fake insertions into dcs, for scraper testing usage::

   python dcssim.py

"""
from datetime import datetime, timedelta
from dcs import Hv, Pw
from dcsconf import DCS
from dcsfake import LCRFaker

import time

def fake_insert(i):
    """
    create Hv instance, fake the LCR attributes and write to dcs
    """
    last = dcs.qd(Hv).first()
    lid = last.id
    assert lid > 0 

    hv = Hv()    
    fk = LCRFaker()
    fk(hv)
    hv.id = lid + 1
    hv.date_time = datetime.now()

    print "%-3d fake_insert %r " % (i, hv) 
    dcs.add(hv)   
    dcs.commit()


if __name__ == '__main__':
    pass
    dcs = DCS("dcs")
    i = 0 
    while i < 10:
        i += 1
        fake_insert(i)
        time.sleep(10)
    print "done"
 
