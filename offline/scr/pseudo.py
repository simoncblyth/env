
class Pseudo(dict):
    """
    pseudo-code translation of the old scraper, in order to see what needs
    to be factored out 
    """
    interval = timedelta(seconds=20)   ## needs to be property of a mapping class ?
    def __init__(self, dcs, off):

        lvld = off.qd(Vld).first()   
        if lvld == None:
            next = 0
        else:
            next = lvld.TIMESTART + self.interval
        self['next'] = next 
        self['recursion'] = 0

        self.dcs = dcs
        self.off = off

    def __call__(self, clsa, clsb ):
        """ 
        TODO:

        #. encapulate the 2 queries into one on a join ... so 
           can tease out the table specifics into one mapped class ... improving 
           genericness

        """ 
        self['recursion'] += 1
        ia = self.dcs.qa(clsa).filter(clsa.date_time >= self['next']).first() 
        if ia == None:
            return
        assert ia.id > 0                

        ib = self.dcs.qa(clsb).filter(clsb.id == ia.id).first()
        if ib == None:
            print "no related clsb %s record for %r " % ( clsb.__name__, ia )
        
        ## this stuff needs to be DybDbied anyhow        
        timeStart = ia.date_time
        timeEnd = timeStart + self.interval
        siteMask = 
        subsite = 

        self['next'] = timeEnd


