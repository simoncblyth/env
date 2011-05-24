import time
from datetime import timedelta

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
    age_threshold = timedelta(hours=2)   ## from the old scraper 7200 seconds

    def __init__(self, sleep):
        self.sleep = sleep

    def changed(self, prev, curr ):
        """
        During a scrape this method is called from the base class, 
        return `True` if the mapping fulfils significant change or age requirements    
        and the propagate method should be invoked.
        If `False` is returned then the propagate method is not called on this iteration of the 
        scraper.

        :param prev: previous source instance
        :param curr: current source instance

        """
        return True

    def propagate(self, curr, contextrange ):
        """
        Override this method in subclasses,  to perform the propagation 
        of the source instance to the target DB. Return True if this 
        succeeds

        :param curr:  current source instance to propagate to target 
        :param contextrange: context range 
        """
        return True

    def __call__(self, max=0):
        """
        Spin the scrape infinite loop, sleeping at each pass.
        Within each pass loop over mappings and check for updated source instances. 
        When updates are found and a previous instance is lodged in the mapping, 
        defer to the subclass to decide if there is significant change to proceed to 
        propagate source instances into target instances.

        NOTE mapping and scrape are intimately entwined        
        """
        i = 0 
        while i<max or max==0:
            i += 1
            for mapping in self:
                update = mapping()
                if not update:              ## no new source instance
                    continue 

                if mapping.prior == None:   ## starting up, need to update 
                    pass
                else:
                    if update.age(mapping.prior) > self.age_threshold:
                        pass   
                    elif self.changed( mapping.prior , update ):
                        pass
                    else:
                        continue

                tcr = update.contextrange( mapping.interval )  
                if self.propagate( update, tcr ):
                    mapping.nexttime = tcr['timeEnd']       
                    mapping.prior = update

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

