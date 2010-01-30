#!/usr/bin/env python
"""
   Usage :
       echo hello | python pipemail.py user@example.com 

     pipe in the content of the message from stdin, the first line 
     will become the subject of the message, and provide the 
     recipient email address as the first argumemnt
"""
 
def sendmail( lines , to , fr="me@localhost" ):
    """
         the first line is used as the subject of the message
         
         creates a text/plain message and sends
         via SMTP server, but does not include the envelope header (?)
    """
    try:
        from email.mime.text import MIMEText
    except ImportError:
        from email.MIMEText import MIMEText

    msg = MIMEText("".join(lines))
    msg['Subject'] = lines[0]
    msg['From'] = fr
    msg['To'] = to

    import smtplib
    s = smtplib.SMTP()
    s.connect()
    print "Attempting to send email to recipient:[%s] from:[%s] message lines:[%s] " % ( to, fr, len(lines) )
    s.sendmail(fr, to, msg.as_string())
    s.close()


if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
    	sendmail( sys.stdin.readlines() , sys.argv[1] )
    else:
	print sys.modules[__name__].__doc__




