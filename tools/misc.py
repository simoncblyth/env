
def getuser():
    """ 
    Alternate to os.getlogin that works when unattached to a terminal,  
    eg from cron or supervisord controlled tasks 
    """
    import os, pwd
    return pwd.getpwuid(os.getuid())[0]
