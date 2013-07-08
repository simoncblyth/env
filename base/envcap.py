#!/usr/bin/env python
"""
Usage::

    envcap.py > ~/env-docs.sh 

TODO:

#. snapshot env, store in pickle and then just do a diff with a new env  

"""
import os
skips = "_ HOME PWD OLDPWD LOGNAME SHELL SHLVL MAIL USER EDITOR TERM DISPLAY SSH_TTY HOSTNAME PS1 SSH_INFOFILE SSH_CLIENT SSH_CONNECTION LS_COLORS SSH_AGENT_PID SSH_AUTH_SOCK".split()


def main():
    kvs = filter(lambda kv:kv[0] not in skips,sorted(os.environ.items(),key=lambda kv:kv[0]))
    print "\n".join(map(lambda kv:"export %s='%s'" % kv ,kvs))

if __name__ == '__main__':
    main()


