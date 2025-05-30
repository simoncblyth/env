#!/usr/bin/env python
"""
 Minimal Eve GUI with ipython shell
"""


def some_callable():
    print "some_callable"

import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
ROOT.PyConfig.GUIThreadScheduleOnce += [ some_callable ]

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    try:
        __IPYTHON__
    except NameError:
        from IPython.Shell import IPShellEmbed
        irgs = ['']
        banner = "entering ipython embedded shell, within the scope of the abtviz instance... try g? for help or the locals() command "
        ipshell = IPShellEmbed(irgs, banner=banner, exit_msg="exiting ipython" )
        ipshell()

