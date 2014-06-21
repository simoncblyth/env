#!/usr/bin/env python

import logging
import numpy as np
log = logging.getLogger(__name__)


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
        log.info("analyse_propagation")
        self.analyse_propagation_last_hit_triangle( propagated )
        self.analyse_propagation_flags( propagated )

    max_slots = property(lambda self:self.dphotons.data.max_slots)

    def analyse_propagation_flags(self, propagated):
        a = propagated
        if a is None:return
        max_slots = self.max_slots
        flags = a['flags'][::max_slots,0]
        t0    = a['flags'][::max_slots,1].view(np.float32)
        tf    = a['flags'][::max_slots,2].view(np.float32)

        log.info("analyse_propagation_flags")
        print "flags", flags
        print "t0", t0
        print "tf", tf
        print "t0 range ", t0.min(), t0.max()
        print "tf range ", tf.min(), tf.max()

        self.time_range = [0., tf.max()]   # start from zero or min ?

    def analyse_propagation_last_hit_triangle(self, propagated):

        a = propagated
        if a is None:return
        max_slots = self.max_slots

        field = 'last_hit_triangle'

        lht = a[field][::max_slots,0]
        photon_id = a[field][::max_slots,1]
        steps = a[field][::max_slots,2]
        slots = a[field][::max_slots,3]

        #assert np.all( lht == -1 )  no longer the case, as are now putting last slot result into slot 0
        assert np.all(np.arange(0,len(photon_id),dtype=np.int32) == photon_id)
        assert np.all( steps == slots )

        counts = np.clip( slots, 0, max_slots-2 ) + 1              # counts of numquad photon records 
        firsts  = np.arange(len(photon_id), dtype='i')*max_slots   # multipled by numquad ?
        assert len(counts) == len(firsts) == len(photon_id)

        # selection like this causes Abort Trap in glDrawArrays call, 
        # except for --debugphoton 0, the first photon
        #
        #if self.config.args.debugkernel and n > 0:
        #    index = np.where( photon_id == self.config.args.debugphoton )[0][0] 
        #    self.counts = counts[index:index+n]
        #    self.firsts = firsts[index:index+n]
        #    self.drawcount = n
        #else:

        self.counts = counts
        self.firsts = firsts
        self.drawcount = len(photon_id)
        pass
        log.info( " counts %s " % str(self.counts))
        log.info( " firsts %s " % str(self.firsts))
        log.info( " drawcount %s " % str(self.drawcount))


if __name__ == '__main__':
    pass

