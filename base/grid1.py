
import pxssh
import os

user=os.environ['USER']
#prompt='\[%s@grid1 %s\]\$ ' % ( user , user )  
prompt=r'\[.*\]\$$\Z' 

print "prompt...%s..." % prompt 

s=pxssh.pxssh()
if not s.login("G1",  user , password="wrong" , terminal_type='ansi', original_prompts=r"\[.*\]\$ ", login_timeout=3 ):
    print "SSH session failed on login."
    print str(s)
else:
    print "SSH session login successful"
    s.sendline ('uname -a')
    print s
    s.prompt()         # match the prompt
    print s.before     # everything before the prompt.
    s.logout()
    print s
                                                               
                                                               


