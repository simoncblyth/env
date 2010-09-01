#!/usr/bin/env python
"""
   Starting point for a Disk usage monitor ...

    TODO :
          * send notification emails when over maxpercent
          * turn into supervisord event listener 
               * http://supervisord.org/events.html#event-listeners-and-event-notifications

"""

import sys, re, os

def write_stdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()

def write_stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

def _main():
    while 1:
        write_stdout('READY\n') # transition from ACKNOWLEDGED to READY
        line = sys.stdin.readline()  # read header line from stdin
        write_stderr(line) # print it out to stderr
        headers = dict([ x.split(':') for x in line.split() ])
        data = sys.stdin.read(int(headers['len'])) # read the event payload
        write_stderr(data) # print the event payload to stderr
        write_stdout('RESULT 2\nOK') # transition from READY to ACKNOWLEDGED


class DiskMon(dict):
    defaults =  dict(disk="/data" , maxpercent="90" , interactive=True, notify="blyth@hep1.phys.ntu.edu.tw" )
    
    _command = "df -h %(disk)s "
    command = property( lambda self:self._command % self )
    
    _percent = re.compile("(\d+)%")
    def get__percent(self):
        l = self._percent.findall( os.popen(self.command).read() )
        return len(l) == 1 and int(l[0]) or 0
    percent = property( get__percent )
    def __repr__(self):
        return "%s \"%s\"  %s   %s " % ( self.__class__.__name__ , self.command , self.percent , dict.__repr__(self)  )


if __name__ == '__main__':
    from optparse import OptionParser
    op = OptionParser()
    ## fix OptionParser making it provide a dict 
    OptionParser.optsdict = property(lambda self: dict([ (k, getattr(self.values,k) ) for k in self.defaults.keys() if hasattr(self.values, k) ]))

    op.add_option("-d", "--disk" )
    op.add_option("-x", "--maxpercent" )
    op.add_option("-n", "--notify" )
    op.add_option("-i", "--interactive" , action="store_true", help="command line non-listener usage for debugging" )

    op.set_defaults( **DiskMon.defaults )    
    op.parse_args() 

    dm = DiskMon( **op.optsdict )



