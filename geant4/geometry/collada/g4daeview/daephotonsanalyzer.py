#!/usr/bin/env python

import logging
import numpy as np
log = logging.getLogger(__name__)

from photons import mask2arg_, count_unique


class DAEPhotonsAnalyzer(object):
    def __init__(self, dphotons):
        self.dphotons = dphotons

    def __call__(self, propagated):
        """
        Interpret counts layed down at tail of propagate_vbo.cu:propagate_vbo

        What an "item" is for glDrawArrays depends on the strides setup in DAEVertexAttrib

        Attempts to draw a single (or few) photon histories are failing, ie
        still see loadsa lines and for everything other than `--debugphoton 0` get
        an Abort Trap::

            g4daeview.sh --with-chroma --load 1 --debugshader --max-slots 10 --debugkernel --debugphoton 0 --debugpropagate 
            g4daeview.sh --with-chroma --load 1 --debugshader --max-slots 10 --debugkernel --debugphoton 1 --debugpropagate 

        So trying alternate means to make vertices disappear by scooting them off to infinity.

        #. NB this is based on a very wasteful and truncating array structure


        """
        log.debug("analyzer.__call__")
        if propagated is None:return

        self.time_range = self.get_time_range( propagated )
        
        history = self.get_history( propagated )
        self.dump_history( history )
        self.history = history 

        counts, firsts, drawcount = self.get_counts_firsts_drawcount(propagated) 

        self.counts = counts
        self.firsts = firsts
        self.drawcount = drawcount 
        pass
        log.debug( " counts %s " % str(self.counts))
        log.debug( " firsts %s " % str(self.firsts))
        log.debug( " drawcount %s " % str(self.drawcount))


    max_slots = property(lambda self:self.dphotons.data.max_slots)

    def get_time_range(self, propagated):
        max_slots = self.max_slots
        flags = propagated['flags'][::max_slots,0]
        t0    = propagated['flags'][::max_slots,1].view(np.float32)
        tf    = propagated['flags'][::max_slots,2].view(np.float32)

        log.debug("analyse_propagation_flags")
        if 0:
            print "flags", flags
            print "t0", t0
            print "tf", tf
            print "t0 range ", t0.min(), t0.max()
            print "tf range ", tf.min(), tf.max()

        time_range = [0., tf.max()]   # start from zero or min ?
        return time_range

    def dump_history(self, history):
        log.info("dump_history")
        present_history_ = lambda _:"%5s %-80s %s " % ( _[0], mask2arg_(_[0]), _[1] )
        print "\n".join(map(present_history_,history))

    def get_history(self, propagated ):
        """
        Counts of all unique history bit settings 
        """ 
        max_slots = self.max_slots
        flags = propagated['flags'][::max_slots,0]
        history = count_unique(flags)  
        return history

    def get_counts_firsts_drawcount(self, propagated):
        """
        Attempts to apply a selection at this level 
        causes Abort Trap in glDrawArrays call, 
        except for --debugphoton 0, the first photon

        ::

            if self.config.args.debugkernel and n > 0:
                index = np.where( photon_id == self.config.args.debugphoton )[0][0] 
                counts = counts[index:index+n]
                firsts = firsts[index:index+n]
                drawcount = n

        """
        max_slots = self.max_slots
        field = 'last_hit_triangle'
        lht = propagated[field][::max_slots,0]
        photon_id = propagated[field][::max_slots,1]
        steps = propagated[field][::max_slots,2]
        slots = propagated[field][::max_slots,3]

        log.debug( " steps %s " % repr(steps))
        log.debug( " slots %s " % repr(steps))
    
        #assert np.all( lht == -1 )  no longer the case, as are now putting last slot result into slot 0
        assert np.all(np.arange(0,len(photon_id),dtype=np.int32) == photon_id)
        if not np.all( steps == slots ):
            log.debug("steps and slots differ\nsteps:%s\nslots:%s" % (repr(steps),repr(slots)))

        counts = np.clip( slots, 0, max_slots-2 )                  # counts of numquad photon records 
        firsts  = np.arange(len(photon_id), dtype='i')*max_slots   # multipled by numquad ?
        drawcount = len(photon_id)
        assert len(counts) == len(firsts) == len(photon_id)
        return counts, firsts, drawcount

if __name__ == '__main__':
    pass

