#!/usr/bin/env python

def sendmail( lines , to , fr="me@localhost" ):
    """
         the first line is used as the subject of the message
         
         creates a text/plain message and sends
         via SMTP server, but does not include the envelope header (?)
    """

    from email.mime.text import MIMEText
    msg = MIMEText("".join(lines))
    msg['Subject'] = lines[0]
    msg['From'] = fr
    msg['To'] = to

    import smtplib
    s = smtplib.SMTP()
    s.connect()
    print "sendmail: to:%s fr:%s lines:%s " % ( to, fr, len(lines) )
    s.sendmail(fr, to, msg.as_string())
    s.close()


if __name__=='__main__':
    """
        pipe in the content of the message, and provide the recipient 
        email address as the first argumemnt
    """
    import sys
    sendmail( sys.stdin.readlines() , sys.argv[1] )





