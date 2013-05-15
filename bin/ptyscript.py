#!/usr/bin/env python
"""
Demo of pty module, Pseudo-terminal utilities
Can use such a pattern as a guardian angel (or daemon) for your bash session.
Everything you type and every response is recorded.

http://docs.python.org/3.2/library/pty.html

::

    [blyth@belle7 ~]$ ptyscript.py 
    Script started, file is typescript
    [blyth@belle7 ~]$ pwd
    /home/blyth
    [blyth@belle7 ~]$ date
    Wed May  8 17:41:37 CST 2013
    [blyth@belle7 ~]$ exit
    exit
    Script done, file is typescript


    [blyth@belle7 ~]$ cat typescript 
    Script started on Wed May  8 17:41:30 2013
    [blyth@belle7 ~]$ pwd
    /home/blyth
    [blyth@belle7 ~]$ date
    Wed May  8 17:41:37 CST 2013
    [blyth@belle7 ~]$ exit
    exit
    Script done on Wed May  8 17:41:40 2013
    [blyth@belle7 ~]$ 



"""
import sys, os, time, getopt
import pty


def main():
    mode = 'wb'
    filename = 'typescript'
    shell = os.environ.get('SHELL','sh')
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'ap')
    except getopt.error, msg:
        print('%s: %s' % (sys.argv[0], msg))
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-a': # append to typescript file
            mode = 'ab'
        elif opt == '-p': # use a Python shell as the terminal command
            shell = sys.executable
    if args:
        filename = args[0]

    script = open(filename, mode)
    scripti = open(filename+"i", mode)

    def master_read(fd):
        data = os.read(fd, 1024)
        script.write(data)
        return data
    def stdin_read(fd):
        data = os.read(fd, 1024)
        scripti.write(data)
        return data

    sys.stdout.write('Script started, file is %s\n' % filename)
    script.write(('Script started on %s\n' % time.asctime()).encode())
    pty.spawn(shell, master_read, stdin_read )
    script.write(('Script done on %s\n' % time.asctime()).encode())
    sys.stdout.write('Script done, file is %s\n' % filename)

if __name__ == '__main__':
    main()


