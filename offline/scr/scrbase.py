import time

class Mapping(object):
    """
    ``Mapping`` instances represent the association between 
    a single table/join in the source DB to a single table/join 
    in the target DB

    Parameters of mapping make most sense to 
    to carried on the target DybDbi class ? and maintained
    as meta entries in the .spec

    #. min interval
    #. max interval
    #. change delta threshold 

    """	
    def __init__(self, source, target, interval):
        """
        Specify source and target SA classes 

        :param source: source SA mapped class (for table/join in non-DBI database)
        :param target: target SA mapped class (for table/join in DBI database)
        :param interval: validity interval `datetime.timedelta` for target DBI writes 

        """
        self.source = source
        self.target = target
        self.interval = interval
        self.next = self.target_nexttime(self.interval)

    def target_nexttime(self, interval):
        """
        Time of last entry in target plus the update interval, 
        which corresponds to earliest source eligibility cutoff
        """
        tl = self.target.last()
        return 0 if tl == None else tl.TIMESTART + interval

    def __call__(self):
        """
        Look for entries in the source with time stamp after the current target `next`. 
        When found return the last instance. 
        """
        eligible = self.source.qafter(self.next).first()   
        if eligible == None:
            print "no src entry after %s " % self.next
            return None
        return eligible


class Scrape(list):
    """
    Base class holding common scrape features 
    """ 
    def __init__(self, sleep):
        self.sleep = sleep
    def __call__(self, max=0):
        """
        Spin the scrape, looping over mappings and calling propagate 
        when eligible source instances are found 
        """
        i = 0 
        while i<max or max==0:
            i += 1
            for mapping in self:
                inst = mapping()
                if not inst:
                    continue 
                tcr = inst.contextrange( mapping.interval )  
                if self.propagate( inst, tcr ):
                    mapping.next = tcr['timeEnd']       
            time.sleep(self.sleep)

class SourceSim(list):
    """
    create fake source instances and insert them 
    """
    def __init__(self, sleep ):
        self.sleep = sleep

    def insertfake(self):
        for source in self:
            last = source.last()
            lid = 0 if last == None else last.id
            lid += 1
            inst = source()
            self.fake( inst , lid )

            print "%-3d insertfake %s %r " % (lid, source, inst) 
            db = source.db
            db.add(inst)   
            db.commit()

    def __repr__(self):
        return "%s %s " % ( self.__class__.__name__, self.sleep )

    def __call__(self, max=10): 
        i = 0 
        while i<max or max==0:
            i += 1
            self.insertfake()
            time.sleep(self.sleep)



if __name__ == '__main__':
    pass

