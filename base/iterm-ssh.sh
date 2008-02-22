
##
## how iTerm sshs with the quick keys 
## the agent-info file is created by the bash function
##     ssh--agent-start 
## that needs to be invoked following reboots 
## this starts the agent and sets up its keys
##
## http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/PasswordLessSSH 
##
source ~/.ssh-agent-info-G
ssh -Y $1

