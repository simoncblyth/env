#!/usr/bin/env python
"""

Former issues, now seemingly resolved:

#. speed changes at some points in interpolation causing jumps
#. view flashes at sawtooth bump points

"""
import logging
log = logging.getLogger(__name__)
import numpy as np


class DAEAnimator(object):
    def __init__(self, period, low=0., high=1.):
        """
        :param period: frame count to go from low to high
        :param low:
        :param high:
        """
        self.count = 0
        self.low = low
        self.high = high
        self.fractions = self.make_fractions(period)
        self._period = period
        self._index = 0
        self._fraction = 0.

    def reset(self):
        self.count = 0 

    def make_fractions(self, period):
        """
        :param period: number of steps between low and high
        """
        return np.linspace(self.low,self.high,num=period)

    nfrac = property(lambda self:len(self.fractions))

    def _get_index(self):
        self._index = self.count % self._period   # modulo, responsible for the sawtooth
        return self._index
    index = property(_get_index)

    def _get_fraction(self):
        index = self.index
        nfrac = self.nfrac
        assert index < nfrac, (index, nfrac) 
        self._fraction = self.fractions[index]
        return self._fraction
    fraction = property(_get_fraction)

    def find_closest_index(self, f ):
        return np.abs(self.fractions - f).argmin()



    def _get_period(self):
        return self._period
    def _set_period(self, period):
        """
        Changing the period adjusts the count in an attempt to prevent 
        jumps in the interpolation fraction. 
        """
        if int(period) == self._period:return
        current_fraction = self.fraction   # do not touch anything until take note of current fraction
        #
        self._period = np.clip(int(period), 25, 10000)
        self.fractions = self.make_fractions(self._period)
        self.count = self.find_closest_index(current_fraction)   
    period = property(_get_period, _set_period)

    def change_period(self, scalefactor):
        self.period = self.period*scalefactor   # setter will adjust fractions and count  

    def _get_bump(self):
        """
        #. bumping at zero rather than at the end of the period avoids interpolation view flashback, 
        #. must avoid count zero however in order to start at first view (rather than second)  
        """ 
        #return self.count % self._period == self._period - 1
        return self.count > 0 and self.count % self._period == 0    
    bump = property(_get_bump)

    def __call__(self):
        """
        :return: value between low and high, bump   

        #. increments only after getting the fraction and bump
        """
        fraction = self.fraction
        bump = self.bump
        self.count += 1            
        return self.low + (self.high-self.low)*fraction, bump
       
    def __repr__(self):
        return "%d/%d" % ( self._index, self._period ) 
 
 
if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)
    animator = DAEAnimator(200)
    for _ in range(40000):
        v = animator()
        print animator
        if _ in (7,18,25):animator.change_period( 0.5 )
        if _ in (50,58,70):animator.change_period( 2.0 )


