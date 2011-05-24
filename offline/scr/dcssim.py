"""
Fake insertions into dcs, for scraper testing usage::
   python dcssim.py

"""
from dcssa import DCS
from scr import PmtHvSim, AdTempSim

if __name__ == '__main__':
    dcs = DCS("dcs")
    #sim = PmtHvSim(dcs, 5 )
    sim = AdTempSim(dcs, 5 )
    print sim 
    sim()



