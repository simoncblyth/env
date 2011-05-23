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
        self.nexttime = self._nexttime(self.interval)
        self.prior = None

    def _nexttime(self, interval):
        """
        TIMESTART of last entry in target plus update validity interval,
        (should be TIMEEND)  
        corresponding to next source cutoff time that will constitute 
        a candidate to be scraped

        Should validity interval exactly match scrape sleep time (the pulse) ?
        """
        tl = self.target.last()
        return 0 if tl == None else tl.TIMESTART + interval


    def __call__(self):
        """
        Look for entries in the source with time stamp after the current `nexttime`. 
        When found return the last instance if the instance meets one of the 
        propagation requirements:

        #. significant delta
        #. significant time since last propagation

        Where does delta-ing belong ?   scraper/mapper/dynamic-class 
        prior clearly belongs in the mapper

        Where do "significant-change" parameters belong ?

        """
        return self.source.qafter(self.nexttime).first()   


class Scrape(list):
    """
    Base class holding common scrape features 
    """ 
    def __init__(self, sleep):
        self.sleep = sleep

    def proceed(self, mapping, update ):
        """
        During a scrape this method is called from the base class, 
        return `True` if the mapping fulfils significant change or age requirements    
        and the propagate method should be invoked.

        If `False` is returned then the propagate method is not called on this iteration of the 
        scraper.
        """
        return True

    def propagate(self, instance, cr ):
        """
        Override this method in subclasses,  to perform the propagation 
        of the source instance to the target DB. Return True if this 
        succeeds

        :param instance:  source instance to propagate to target 
        :param cr: context range 
        """
        return True


    def __call__(self, max=0):
        """
        Spin the scrape, looping over mappings and calling propagate 
        when eligible source instances are found 
        """
        i = 0 
        while i<max or max==0:
            i += 1
            for mapping in self:
                instance = mapping()
                if not instance:
                    continue 
                if not self.proceed( mapping , instance ):
                    continue
                tcr = instance.contextrange( mapping.interval )  
                if self.propagate( instance, tcr ):
                    ## NOTE diddling with mapping attribute directly ... mapping and scrape are intimately entwined        
                    mapping.nexttime = tcr['timeEnd']       
                    mapping.prior = instance
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

