import time
from datetime import timedelta

import logging
log = logging.getLogger(__name__)


class Mapping(object):
    """
    ``Mapping`` instances represent the association between 
    a single table/join in the source DB to a single table/join 
    in the target DB

    """	
    def __init__(self, source, target, interval):
        """
        Specify source and target SA classes, determines ``nexttime`` for 
        a candidate source instance by looking at target instances

        :param source: source SA mapped class (for table/join in non-DBI database)
        :param target: target SA mapped class (for table/join in DBI database)
        :param interval: validity interval `datetime.timedelta` for target DBI writes 

        """
        self.source = source
        self.target = target
        self.interval = interval
        self.nexttime = self.nexttime_(self.interval)
        self.prior = None
        self.counter = -1
        log.info( "create mapping %s " % self )

    def __str__(self):
        return "(%3d) %s : %s : nextt %s interval %s " % ( self.counter, self.source.__name__, self.target.__name__, self.nexttime, self.interval )

    def nexttime_(self, interval):
        """
        TIMESTART of last entry in target plus update validity interval (normally would be TIMEEND)  
        corresponding to next source cutoff time that will constitute a candidate to be scraped
        Should validity interval exactly match scrape sleep time (the pulse) ?
        """
        tl = self.target.last()
        return 0 if tl == None else tl.TIMESTART + interval


    def __call__(self, counter=0):
        """
        Returns last source instance with timestamp after current `nexttime`. 

        Q:

        #. should this be playing catchup, when a scraper is restarted after hiatus ?
        """
        self.counter = counter
        return self.source.qafter(self.nexttime).first()   



class Player(list):
    def __init__(self, sleep):
        self.sleep = sleep

class Scraper(Player):
    """
    Base class holding common scrape features 
    """ 
    age_threshold = timedelta(hours=2)   ## from the old scraper 7200 seconds

    def __init__(self, sleep):
        Player.__init__(self, sleep)

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
                update = mapping(i)         ## i : counter for debugging 
                if not update:              ## no new source instance
                    log.info("no update for %s " % mapping )
                    continue 

                if mapping.prior == None:   ## starting up, need to update 
                    log.info("no prior for %s " % mapping )
                    pass
                else:
                    age = update.age(mapping.prior) 
                    if age > self.age_threshold:
                        log.info("update age %r over threshold %s for : %s " % (age, self.age_threshold, mapping) )
                        log.info("date_time for prior %s and update %s " % (mapping.prior.date_time, update.date_time ))
                        pass   
                    elif self.changed( mapping.prior , update ):
                        log.info("significant change detected %s " % mapping )
                        pass
                    else:
                        log.info("skip update %s " % mapping )
                        continue

                tcr = update.contextrange( mapping.interval )  
                if self.propagate( update, tcr ):
                    log.debug("propagation succeeded %s " % mapping )
                    mapping.nexttime = tcr['timeEnd']       
                    mapping.prior = update
                else:
                    log.warn("propagation failed %s " % mapping )
                  

            time.sleep(self.sleep)

class Faker(Player):
    """
    create fake source instances and insert them 
    """
    def __init__(self, sleep):
        Player.__init__(self, sleep)

    def insertfake(self):
        for source in self:
            last = source.last()
            lid = 0 if last == None else last.id
            lid += 1
            inst = source()
            self.fake( inst , lid )

            log.info("%-3d insertfake %s %r " % (lid, source.__name__, inst.asdict ))
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

