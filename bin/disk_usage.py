#!/usr/bin/env python
from env.mysqlhotcopy.fsutils import disk_usage 
if __name__ == '__main__':
    du = disk_usage()
    print du.asdict()

