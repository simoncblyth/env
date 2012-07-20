#!/usr/bin/env python
from env.tools.sendmail import sendmail
sendmail("subject\nbody line 1\nbody line 2\n".split("\n"), "blyth@hep1.phys.ntu.edu.tw" )

