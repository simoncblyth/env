#!/usr/bin/env python
"""
   Usage :
       echo hello | python pipemail.py user@example.com 

     pipe in the content of the message from stdin, the first line 
     will become the subject of the message, and provide the 
     recipient email address as the first argumemnt
"""
import os, socket, logging
log = logging.getLogger(__name__)

def sendmail( text, to , fr=os.environ.get('FROM',"me@localhost"), delim="\n" ):
    """


         the first line is used as the subject of the message
         
         creates a text/plain message and sends
         via SMTP server, but does not include the envelope header (?)
    """
    try:
        from email.mime.text import MIMEText
    except ImportError:
        from email.MIMEText import MIMEText

    lines = text.split(delim)
    msg = MIMEText(delim.join(lines))
    msg['Subject'] = lines[0]
    msg['From'] = fr
    msg['To'] = to

    import smtplib
    s = smtplib.SMTP()
    try:
        s.connect()
        log.info("Attempting to send email to recipient:[%s] from:[%s] message lines:[%s] " % ( to, fr, len(lines) ))
        s.sendmail(fr, to, msg.as_string())
    except socket.error, se:
	log.warn("socket.error while attempting to sendmail : %s " % se  )  
    finally:
        s.close()

if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
    	sendmail( sys.stdin.read() , sys.argv[1] )
    else:
	print sys.modules[__name__].__doc__




