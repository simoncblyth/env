#!/usr/bin/env python
"""
 
   Monitors disk usage percentage, and sending notification
   emails if usage exceeds limits, 
   typically invoked daily/hourly via a cron command line :
     
       python %(path)s --disk=/home --maxpercent=85 --mailto=jack@%(node)s,jill@%(node)s

"""
import sys, re, os, platform, logging, getpass

log = logging.getLogger("diskmon")
sth = logging.StreamHandler()
sth.setFormatter(logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s"))
log.addHandler(sth)


ctx = dict( user=getpass.getuser(), node=platform.node(), path=sys.argv[0], basename=os.path.basename(sys.argv[0]) )


class Mail(str):
    """
        This primitive mailer is now only used with the --primitive option  
    """ 
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
    defaults =  dict(
                     disk="/home" , 
                 loglevel="INFO", 
               maxpercent="90" , 
                  dry_run=False , 
                      msg="not-checked",
                  primitive=False,
                   mailto="%(user)s@%(node)s" % ctx , 
                  mailhost="localhost",
                  mailfrom="%(basename)s@%(node)s" % ctx,
                 mailsubj="%(basename)s on node %(node)s" % ctx,
                  mailbuffer=10
                 )

    def create(cls):
        from optparse import OptionParser
        op = OptionParser(usage="./%prog [options] " + __doc__ % ctx )

        op.add_option("-d", "--disk"       , help="path to be monitored by \"df -h\", default : %default ")
        op.add_option("-x", "--maxpercent" , help="maximum allowed percentage , default : %default " )
        op.add_option("-e", "--mailto"     , help="comma delimited list of email addresses, default : %default " )
        op.add_option("-l", "--loglevel"   , help="logging level : INFO, DEBUG, WARN etc.. , default : %default " )
        op.add_option("-n", "--dry-run"    , action="store_true" , help="do not send emails, for debugging, default : %default " )
        op.add_option("-p", "--primitive"  , action="store_true", help="use primitive sendmail, default : %default ") 
        op.add_option("-o", "--mailhost"   , help="smtp host, default : %default " )
        op.add_option("-f", "--mailfrom"   , help="smtp from address, default : %default " )
        op.add_option("-b", "--mailbuffer" , help="max number of log messages per mail, default : %default " )
        op.add_option("-j", "--mailsubj"   , help="mail subject, default : %default " )

        
        op.set_defaults( **DiskMon.defaults )
        opts,args = op.parse_args() 
        
        loglevel = getattr(logging,opts.loglevel.upper()) 
        log.setLevel(loglevel)
        sth.setLevel(loglevel)

        if not opts.primitive and not opts.dry_run:
            from buffering_smtp_handler import BufferingSMTPHandler
            bsh = BufferingSMTPHandler( opts.mailhost % ctx, opts.mailfrom % ctx, opts.mailto.split(",") , opts.mailsubj % ctx,  opts.mailbuffer )
            bsh.setLevel(loglevel)
            log.addHandler(bsh)

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
        if not self.inlimit:
            self.update(msg="percentage %s EXCEEDS LIMIT" % self.percent ) 
            log.error( self ) 
            if self["primitive"]:Mail(self)(self["mailto"])
        else:
            self.update(msg="percentage %s within limit" % self.percent ) 
            log.debug( self ) 


if __name__ == '__main__':
    dm = DiskMon.create()
    dm()


