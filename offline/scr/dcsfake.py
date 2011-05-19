import re

class Faker(object):
    """
    Sets fake attributes for an object, by matching object
    attribute names to patterns  
    """
    def __call__(self, obj , Fkr ):  
        fkr = Fkr()
        for k in filter(lambda _:not _.startswith("_"),dir(obj)):
            if fkr.match(k):
                setattr( obj , k , fkr.fake )

class LCR(dict):
    kptn = re.compile("^L(?P<l>\d)C(?P<c>\d)R(?P<r>\d)$")
    attr = property(lambda self:"L%(l)sC%(c)sR%(r)s" % self )
    fake = property(lambda self:int(self['l'])*100 + int(self['c'])*10 + int(self['r'])) 
    def match(self, k):
        m = self.kptn.match(k)
        if m:
            self.update( m.groupdict() )
            return True
        return False 

class LCRFaker(Faker):
    """
    Specialization of Faker to ladder/ring/column mapped objects, usage::

         lf = LCRFaker()
         lf(obj)
         obj.id = 1
         obj.date_time = datetime.now() 
 
    """
    def __call__(self, obj):
        Faker.__call__( self, obj , LCR  ) 

class Demo(object):
    pass



if __name__=='__main__':
    pass

    d = Demo()
    atts = "L1C1R1 L1C1R2 L1C1R3".split()
    for att in atts:
       setattr( d, att, 0) 

    lf = LCRFaker()
    lf(d)

    for att in atts:
       v = getattr(d, att)
       print att, v 




