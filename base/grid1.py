
import pxssh
import os

user=os.environ['USER']

dbg=0

s=pxssh.pxssh()
if not s.login("G1",  user , password="wrong" , terminal_type='ansi', original_prompts=r"\[.*\]\$ ", login_timeout=3 ):
    print "SSH session failed on login."
    print str(s)
else:
    print "SSH session login successful"
    s.sendline ('uname -a')
    
    if dbg > 0:
        print s
    
    s.prompt()         # match the prompt
    print s.before     # everything before the prompt.
    
    s.logout()
    
    if dbg>0:
        print s
                                                               
                                                               


