#!/usr/bin/env python
"""
Raw demo of whats happening behind the scenes

"""
import sys

def write_stdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()
def write_stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()
def main():
    while 1:
        write_stdout('READY\n') # transition from ACKNOWLEDGED to READY
        line = sys.stdin.readline()  # read header line from stdin
        write_stderr("\nheader[%s]" % line.strip()) # print it out to stderr
        headers = dict([ x.split(':') for x in line.split() ])
        data = sys.stdin.read(int(headers['len'])) # read the event payload
        write_stderr("\npayload[%s]" % data) # print the event payload to stderr
        write_stdout('RESULT 2\nOK') # transition from READY to ACKNOWLEDGED

if __name__ == '__main__':
    main()


