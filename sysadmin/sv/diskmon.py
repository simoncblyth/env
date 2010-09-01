#!/usr/bin/env python
"""
   Starting point for a Disk usage monitor ...

    TODO :
          * hostname, date identification in notifications 
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

class Mail(str):
    sendmail = '/usr/sbin/sendmail -t -i'
    def __call__(self, email ):
        body =  'To: %s\n' % email
        body += 'Subject: %s\n' % self.split("\n")[0]
        body += '\n'
        body += self
        m = os.popen(self.sendmail, 'w')
        m.write(body)
        m.close()

class DiskMon(dict):
    """
        Check disk usage percentage and send notification emails 
    """
    defaults =  dict(disk="/data" , maxpercent="90" , interactive=True, notify="blyth@hep1.phys.ntu.edu.tw" , msg="DEFMSG" )

    _command = "df -h %(disk)s "
    _percent = re.compile("(\d+)%")
    _warning = """DiskMon %(disk)s %(maxpercent)s %(msg)s"""

    command = property( lambda self:self._command % self )
    percent = property( lambda self:self._percent.findall( os.popen(self.command).read() )[0] )
    warning = property( lambda self:self._warning % self )
    
    def __call__(self):
        p = self.percent
        ok = p < int(self['maxpercent']) 
        self.update( msg = (" percentage %s EXCEEEDS LIMIT ", " percentage %s within limit " )[ok] % p ) 
        Mail( self.warning )( self['notify'] )    


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
    dm()


