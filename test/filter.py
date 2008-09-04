"""
  http://www.oreillynet.com/onlamp/blog/2007/08/pymotw_subprocess_1.html
  http://blog.doughellmann.com/2007/07/pymotw-subprocess.html

  reads on stdin and writes to stdout 

"""

import sys
sys.stderr.write('filter.py: starting\n')

while True:
    next_line = sys.stdin.readline()
    if not next_line:
        break
    sys.stdout.write(next_line)
    sys.stdout.flush()

sys.stderr.write('filter.py: exiting\n')




