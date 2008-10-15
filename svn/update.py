class SVNUpdate(dict):
    """
     
        from env.svn import SVNUpdate
        up = SVNUpdate("/Users/blyth/env")
        up.update()
        for p in up("A"):
            print p
        
            
    """
    def __init__(self, dir ):
        self.dir = dir
        
    def logpath(self, rev="" ):
        import os
        return os.path.join(self.dir, "svnupdate-%s.log" % rev )
        
    def update(self, rev=None ):
        """
           Supplying a revision, implies to read a former update log 
           not giving a revision, does the update, parses and writes the log
        """
        if rev:
            logp = self.logpath(rev)
            import os
            if os.path.exists(logp):
                print "reading/parsing the logfile %s " % logp
                log = file(logp).readlines()
                self.parse(log)
            else:
                print "path %s does not exist " % logp 
            return
        
        import subprocess
        cmd = 'cd %s ; svn up ' % self.dir
        prc = subprocess.Popen( cmd , shell=True , stdout=subprocess.PIPE , stderr=subprocess.PIPE  )
        out, err = prc.communicate()
        if prc.returncode == 0:
            self.parse( out.split("\n"))
            logp = self.logpath(self.revision)
            print "writing update log to %s " % logp 
            f = open(logp, "w" )
            f.write( out )
            f.close()
        else:
            self['cmd'] = cmd
            self['returncode'] = prc.returncode 
            self['out'] = out
            self['error'] = err
            print "error %s while doing %s " % ( prc.returncode, cmd )
            print "out %s err %s " % ( out , err )


    def parse(self, lines):
        """
        cat /usr/local/dyb/trunk_dbg/NuWa-trunk/dybgaudi/up.log | python update.py
        python update.py /usr/local/dyb/trunk_dbg/NuWa-trunk/dybgaudi/up.log
          
svn up
A    Simulation/DetSim/python/DetSim
A    Simulation/DetSim/python/DetSim/__init__.py
A    Simulation/DetSim/python/DetSim/Default.py
D    Simulation/DetSim/share/far.py
D    Simulation/DetSim/share/allPhysics.py
D    Simulation/DetSim/share/nearDB.py
D    Simulation/DetSim/share/nearLA.py
D    Simulation/DetSim/share/basicPhysics.py
D    Simulation/DetSim/share/default.py
U    Simulation/DetSim/cmt/requirements
A    Simulation/GenTools/python/GenTools
A    Simulation/GenTools/python/GenTools/Examples.py
A    Simulation/GenTools/python/GenTools/__init__.py
D    Simulation/GenTools/share/default.py
D    Simulation/GenTools/share/electron_gun.py
D    Simulation/GenTools/share/addDumper.py
U    Simulation/GenTools/cmt/requirements
D    Simulation/SimuAlg
U    Simulation/ElecSim/src/components/EsPmtEffectPulseTool.cc
U    Simulation/ElecSim/src/components/EsIdealPulseTool.cc
UU   DybPython/python/DybPython/Control.py
U    DybPython/python/DybPython/ParseArgs.py
...
A    RootIO/PerTrigEvent/PerTrigEvent/PerTrigCommandCollection.h
Updated to revision 4343.
        """
        
        import re
        
        rev = re.compile('Updated to revision (?P<revision>\S*)\.')
        self.revision = None
        for i in (1,2,3):
            line = lines[-i]
            print "try line %s " % line  
            m = rev.match(line)
            if m:
                self.revision = m.group('revision')
                break
        assert self.revision 
        
        print "last line is %s " % i 
        
        
        att = re.compile('(?P<status>\S*)\s*(?P<path>\S*)')
        for line in lines[:-i]:
            m = att.match(line)
            if m:
                status, path = m.group('status'), m.group('path')
                self[path] = status
            else:
                print "did not match ... %s " % line 
            
    def __repr__(self ):
        return "<SVNUpdate %s %s >" % (self.dir, self.revision )
        
    def categories(self):
        return list(set(self.values()))
    
    def paths(self, s , absolute=False, onlydir=False, onlyfile=False ):
        """
            note if onlydir and onlyfile are True, the onlydir wins 
        """
        import os
        iwd = os.getcwd()
        os.chdir( self.dir )
        
        if absolute:
            prep = lambda p:os.path.join(self.dir, p)
        else:
            prep = lambda p:p
            
        if onlydir:
            sele = lambda p:os.path.isdir(p)
        elif onlyfile:
            sele = lambda p:os.path.isfile(p)
        else:
            sele = lambda p:True
            
        l = sorted([prep(p) for p in self.keys() if self[p] == s and sele(p)])
        os.chdir( iwd )
        return l

    def __call__(self, c , **kwa ):
        return self.paths(c, **kwa)
        

if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        log = file(sys.argv[1]).readlines()
    
    up = SVNUpdate('/usr/local/dyb/trunk_dbg/NuWa-trunk/dybgaudi')
    up.update(rev=4343)   ## give a rev to reread prior log
    #up.update()     ## default ... do the update
    
    print up
    for c in up.categories():
        for onlydir in (False,True):
            print "=== category %s  onlydir %s " % (c, onlydir)
            for p in up(c, absolute=False, onlydir=onlydir ):
                print "%s " % p 
    
