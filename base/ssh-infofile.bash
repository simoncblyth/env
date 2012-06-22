
ssh-infofile-usage(){ cat << EOU

EOU
}


 SSH_INFOFILE=$HOME/.ssh-agent-info-$NODE_TAG
 export SSH_INFOFILE
 [ -r $SSH_INFOFILE ]   && . $SSH_INFOFILE

