#!/usr/bin/env python
"""

"""

import logging
log = logging.getLogger(__name__)
from env.plot.highmon import HighMon

class TGZMon(HighMon):
    def __init__(self, *args, **kwa ):
        HighMon.__init__(self, *args, **kwa )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    url = "http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json"
    mon = TGZMon(url)
    mon()



