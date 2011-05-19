"""
Fake insertions into dcs, for scraper testing usage::

   python dcssim.py

"""
from datetime import datetime, timedelta
from dcsconf import DCS
from dcsfake import LCRFaker

import time

def fake_insert(i):
    """
    create generic instances, fake the LCR attributes and write to dcs
    """

    for gcln, gcls in dcs.classes.items():
        last = dcs.qd(gcls).first()
        if last == None:
            lid = 0
        else:
            lid = last.id

        inst = gcls()    
        fk = LCRFaker()
        fk(inst)
        inst.id = lid + 1
        inst.date_time = datetime.now()

        print "%-3d fake_insert %s %r " % (i, gcln, inst) 

        dcs.add(inst)   
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
 
