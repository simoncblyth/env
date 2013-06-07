#!/usr/bin/env python
import sys
from env.mysqlhotcopy.fsutils import DiskUsage
if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = None
    du = DiskUsage(path=path)
    print du.asdict()

