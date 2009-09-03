

def path_():
    import os
    return os.path.join( os.path.dirname(__file__), 'test_Conventions.ref' )



from vdbi import VLD_COLUMNS


class Enum(dict):
    def __init__(self,path=None):
        if not(path):path=path_()
        self.update(eval(file(path).read()))
        self._defaults()
    
    def __call__(self, name , v ):
        assert name in self.keys()
        if type(v) == str:
            return self[name].get(v, None)
        elif type(v) == int:
            if type(self[name]) == str:
                return v
            else:
                for k,i in self[name].items():
                    if v == i:return k
        return None

    def options(self, s ):
        return [(v,k) for k,v in sorted( self[s].items() , key=lambda (a,b):(b,a) ) ]

    def _defaults(self):
        default = { 
                   'Site':self('Site','kDayaBay'), 
                'SimFlag':self('SimFlag','kMC') , 
              'DetectorId':self('DetectorId','kUnknown') }
        
        n2a = {       'Site':VLD_COLUMNS['SITEMASK'], 
                   'SimFlag':VLD_COLUMNS['SIMMASK'], 
                'DetectorId':VLD_COLUMNS['SUBSITE'], 
                 'TimeStart':VLD_COLUMNS['TIMESTART'],
                   'TimeEnd':VLD_COLUMNS['TIMEEND'], }
        a2n = {}
        for n,a in n2a.items():
            if n.startswith('Time'):n = 'Timestamp'
            a2n[a] = n        
        
        self['_default']  = default
        self['_name2attr'] = n2a
        self['_attr2name'] = a2n


ctx = Enum()



if __name__=='__main__':
    from vdbi.dyb import ctx

    for t in ctx.keys():
        if not(hasattr(ctx[t], 'items')):
            print "simple.. %s %s " % ( t , ctx[t] )
        else:
            for n,i in ctx[t].items():
                i = ctx( t , n ) 
                k = ctx( t , i )
                assert k == n
                print t,n,i,k

    print ctx('Site','kDayaBay')
    print ctx('Site',1)

    assert ctx('SimFlag','kMC') == 2
    assert ctx('SimFlag', 2   ) == 'kMC'





   
