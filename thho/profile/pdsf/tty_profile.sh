#!/bin/bash
# tty_profile
################### prompt etc. setting ############################

WHITE="\[\033[1;37m\]"
LIGHT_BLUE="\[\033[1;34m\]"
NO_COLOUR="\[\033[0m\]"

PS1="$WHITE[\u@\h \w]#\#\
\n\
\$ $NO_COLOUR"

export PS1

################### alias setting #####################
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
alias ll='ls -alh --color=auto'
alias ls='ls --color=auto'
