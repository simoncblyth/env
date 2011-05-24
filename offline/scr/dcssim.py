"""
Fake insertions into dcs, for scraper testing usage::
   python dcssim.py

"""
from dcssa import DCS
from scr import PmtHvFaker, AdTempFaker

if __name__ == '__main__':
    dcs = DCS("dcs")
    #sim = PmtHvFaker(dcs, 5 )
    sim = AdTempFaker(dcs, 5 )
    print sim 
    sim()



