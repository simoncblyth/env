"""

   ipython leakr.py MDC_runLED_Hephaestus.log 

"""

import re



class Leak(dict):
    ls = []
    patn={
       'smry':re.compile('^Leak summary for'),
      'leaks':re.compile('^(?P<nleak>\d*)\s*leaks\s*\(total\s*(?P<total>\d*)\s*bytes\), originating in:'),
      'bytes':re.compile('^(?P<identifier>.*)\s*\((?P<bytes>\d*) bytes\)\s*$'),
      'stack':re.compile('^\s*(?P<addr>0x\S*)\s*(?P<frame>.*)$')
    }

    @classmethod
    def parse( cls, path='MDC_runLED_Hephaestus.log' ):
        
        lines = []
        for line in file(path).readlines():
            for n,p in cls.patn.items():
                m = p.match( line )
                if m:
                    if n in ('leaks','smry'):
                        ## collect the prior chunk and start a new one 
                        if len(lines)>0:
                            lk = Leak() ; 
                            cls.ls.append(lk)
                            lk.lines = lines    
                        lines = []
                    lines.append(line)    ## collected matched lines

        ## now over the chunks
        for lk in cls.ls:
            for line in lk.lines:
                for n,p in cls.patn.items():
                    m = p.match( line )
                    if m:
                        if n=='stack':
                            lk.addlist( 'frame' , m.group('frame') )
                            lk.addlist( 'addr' ,  m.group('addr') )
                        else: 
                            for k,v in m.groupdict().items():
                                if k in ('total','nleaks','bytes'):
                                    lk.update( {k:int(v)} )
                                else:
                                    lk.update( {k:v} )
 
            if len(lk.keys()) != 6:
                for l in lk.lines:
                    print "MIS-PARSE:%s" % l  



    def addlist( self, k, v ):
        if not(v):
            return
        if not(self.has_key(k)):
            self[k] = []
        self[k].append(v)


    def __repr__(self):
        return "Leak({" + "".join( [" %s:'%s' " % ( k, self.get(k,'?') ) for k in ("total", "nleak", "bytes", "identifier", "frame","addr" ) ]) + "})" 

    def __str__(self):
        bt = '' 
        if len(self['addr']) == len(self['frame']):
            for a,f in zip(self['addr'], self['frame']):
                bt += "   %s %s\n" % ( a, f )
        self['bt'] = bt 
        return """
%(nleak)s leaks (total %(total)s bytes), originating in: 
%(identifier)s (%(bytes)s bytes) 
%(bt)s 
""" % dict(self)


if __name__=='__main__':
    import sys
    if len(sys.argv) < 2 :
        print __doc__
    else:
        Leak.parse(sys.argv[1])
        Leak.ls.sort(key=lambda x:x['total']  )
        for lk in Leak.ls:
            if lk['total'] > 10000:
                print lk
                #for l in lk.lines:
                #    print l,

