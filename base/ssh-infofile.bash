
ssh-infofile-usage(){ cat << EOU

Hmm, why is this split off ?

EOU
}


#SSH_INFOFILE=$HOME/.ssh-agent-info-$NODE_TAG
SSH_INFOFILE=$HOME/.ssh-agent-info
export SSH_INFOFILE
[ -r $SSH_INFOFILE ]   && . $SSH_INFOFILE

