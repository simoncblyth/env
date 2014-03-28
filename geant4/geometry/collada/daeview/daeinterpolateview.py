#!/usr/bin/env python
"""
"""
import logging, math
log = logging.getLogger(__name__)
import numpy as np

class DummyView(object):
    def __init__(self, j):
        self.j = j 
    def __repr__(self):
        return "%s %s " % (self.__class__.__name__, self.j )
    eye_look_up = property(lambda self:self.j*np.arange(1,10))

def oscillate( frame, low, high, speed ):
    return low + (high-low)*(math.sin(frame*math.pi*speed)+1.)/2. 

def sawtooth( frame, low, high, speed ):
    p = int(1./speed)
    return low + (high-low)* float(frame%(p+1))/float(p) 


class DAEInterpolateView(object):
    """
    could swap `current_view` as f > 0.5, no too confusing 
    keep current as the "A" view, 
    only moving to the next view at the next cycle 
    (when the original "B" view becomes the "A" view of the next cycle)

    How to change animation speed without changing position 
    in the interpolation ?

    """
    def __init__(self, views):
        assert len(views) > 1
        self.views = views
        self.cycle = 0
        self.define_pair(0,1)
        self.f = 0.

    nviews = property(lambda self:len(self.views))  # could add a view during operation ?

    current_view = property(lambda self:self.views[self.i])   
    next_view = property(lambda self:self.views[self.j])

    # pass along non-interpolated attribute access to current view, ie the "A" view
    solid = property(lambda self:self.current_view.solid)
    extent = property(lambda self:self.current_view.extent)
    target = property(lambda self:self.current_view.target)
    index = property(lambda self:self.current_view.index)
    model2world = property(lambda self:self.current_view.model2world)
    world2model = property(lambda self:self.current_view.world2model)

    def _get_eye_look_up(self):
        """Linear interpolation of two 9 element arrays """
        return (1.-self.f)*self.a + self.f*self.b
    eye_look_up = property(_get_eye_look_up)     

    def __repr__(self):
        return " ".join([
                         "IV(%d->%d[%s] %d)" % (self.i,self.j,self.f,self.nviews), 
                         "A %s" % self.views[self.i],
                         "B %s" % self.views[self.j],
                        ])  

    def define_pair(self, i , j ):
        self.i = i  
        self.j = j
        log.info("define_pair cycle %s i %s j %s : %s %s " % (self.cycle, i,j,  self.views[i], self.views[j])  )
        self.a = self.views[i].eye_look_up
        self.b = self.views[j].eye_look_up 
        self.cycle += 1

    def next_cycle(self):
        i = (self.i + 1) % self.nviews           
        j = (self.j + 1) % self.nviews            
        self.define_pair(i,j )

    def __call__(self, count, speed):
        """
        For i = 0, as f reaches 1, 
        the interpolation A->B is complete and reach v1. 

             A->B    v0->v1
             0  1

        The sawtooth is about to plunge f back to 0, so 
        next cycle changes A and B to become v1->v2

        i = 1
             A->B    v1->v2 
             1  2  
   
        """
        f = sawtooth( count, 0., 1., speed )
        bump = np.allclose(f,1.)
        #log.info("count %s f %s speed %s bump %s " % ( count, f, speed, bump ))
        self.f = f
        if bump:self.next_cycle()
            

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)


    nview = 2
    views = map(DummyView, np.arange(0,nview))
    print views

    view = DAEInterpolateView( views) 

    for frame in range(100):
        view(frame, 0.1)        
        #print view
        




    

