#!/usr/bin/env python
"""


"""
import sys, re, os
import logging

log = logging.getLogger("diskmon")
hdl = logging.StreamHandler()
hdl.setFormatter(logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s"))
log.addHandler(hdl)

class Mail(str):
    sendmail = '/usr/sbin/sendmail -t -i'
    def send(self, email ):
        body =  'To: %s\n' % email
        body += 'Subject: %s\n' % self.split("\n")[0]
        body += '\n'
        body += self
        m = os.popen(self.sendmail, 'w')
        m.write(body)
        m.close()

    def __call__( self , recipients ):
        for recipient in recipients.split(","):
            self.send( recipient )    

class DiskMon(dict):
    """
        Check disk usage percentage and send notification emails 
    """
    defaults =  dict(disk="/data" , loglevel="INFO", maxpercent="90" , dry_run=True , notify="blyth@hep1.phys.ntu.edu.tw" , msg="not-checked" )
    def create(cls):
        from optparse import OptionParser
        op = OptionParser(usage="./%prog [options] ")
        op.add_option("-d", "--disk"       , help="path to be monitored by \"df -h\", default : %default ")
        op.add_option("-x", "--maxpercent" , help="maximum allowed percentage , default : %default " )
        op.add_option("-e", "--notify"     , help="comma delimited list of email addresses, default : %default " )
        op.add_option("-l", "--loglevel"   , help="logging level : INFO, DEBUG, WARN etc.. , default : %default " )
        op.add_option("-n", "--dry-run"    , action="store_true" , help="do not send emails, for debugging, default : %default " )
        op.set_defaults( **DiskMon.defaults )
        opts,args = op.parse_args() 
        
        loglevel = getattr(logging,opts.loglevel.upper()) 
        log.setLevel(loglevel)
        hdl.setLevel(loglevel)
        return cls(vars(opts))
    create = classmethod(create)
 
    _command = "df -h %(disk)s "
    command = property( lambda self:self._command % self )

    def get_percent(self):
        if not self.has_key('percent'):
            self['percent'] = re.compile("(\d+)%").findall( os.popen(self.command).read() )[0] 
        return self['percent']
    percent = property( get_percent )
    inlimit = property( lambda self:self.percent < self['maxpercent'] )
    
    def __repr__(self):
        return """DiskMon %(disk)s : %(msg)s : %(maxpercent)s""" % self

    def __call__(self):
        log.debug( self )
        if not self.inlimit:
            self.update(msg="percentage %s EXCEEDS LIMIT" % self.percent ) 
            log.error( self ) 
            Mail(self)(self["notify"])
        else:
            self.update(msg="percentage %s within limit" % self.percent ) 
            log.debug( self ) 


if __name__ == '__main__':
    dm = DiskMon.create()
    dm()


