#!/usr/bin/env python
"""
usage: %prog REPOS TXN

Based on example Subversion pre-commit hook from 

* http://wordaligned.org/articles/a-subversion-pre-commit-hook

testing the pre-commit hook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create dummy repo::

    mkdir -p /tmp/repos
    svnadmin create /tmp/repos/dummy
    ls -l /tmp/repos/dummy/hooks

Plant the hook::

    #cp $ENV_HOME/svn/pre-commit-disallow-tabs.py /tmp/repos/dummy/hooks/pre-commit && chmod ugo+x  /tmp/repos/dummy/hooks/pre-commit
    #ln -s ~/e/svn/pre-commit-disallow-tabs.py pre-commit  # actually try with symbolic link 
    ln /Users/blyth/e/svn/pre-commit-disallow-tabs.py pre-commit      # try hard link

Checkout working copy::

    simon:e blyth$ mkdir -p /tmp/wc && cd /tmp/wc
    simon:wc blyth$ svn co file:///tmp/repos/dummy 
    Checked out revision 0.

Try to commit::

    simon:dummy blyth$ svn add hello.py 
    A         hello.py
    simon:dummy blyth$ ci -m "first commit is tab clean "
    Adding         hello.py
    Transmitting file data .svn: E165001: Commit failed (details follow):
    svn: E165001: Commit blocked by pre-commit hook (exit code 255) with no output.

Try the test technique::

    python $ENV_HOME/svn/pre-commit-disallow-tabs.py -r /tmp/repos/dummy 1

Argh still at rev 0::

    simon:dummy blyth$ python $ENV_HOME/svn/pre-commit-disallow-tabs.py -r /tmp/repos/dummy 1
    svnlook: E160006: No such revision 1

Succeed to commit after fixing shebang line, hmm failing to prevent a tab::

    simon:dummy blyth$ ci -m "try to commit python with a tab  "
    Sending        hello.py
    Transmitting file data .
    Committed revision 3.
    simon:dummy blyth$ 

Hmm the hook is noting the tab but the commit it not being stopped::

    simon:hooks blyth$ ./pre-commit --revision .. 5
    INFO:__main__:svnlook %s .. --revision 5 
    WARNING:__main__:tab detected in world.py 
    simon:hooks blyth$ 

Presumably the pre-commit command runs in some funny environment ?


"""
import os, sys, subprocess, logging
log = logging.getLogger(__name__)

def command_output(cmd):
    "Capture a command's standard output."
    return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()[0]

def au_files(look_cmd, exts):
    """ 
    :param look_cmd: with txn or revision specification
    :param exts: space delimited list of file extensions, eg ".cpp .cxx .h"
    :return: list of added or updated files

    `svnlook changed` gives output like::

       U   trunk/file1.cpp
       A   trunk/file2.cpp
    """
    exts = exts.split()
    changed = command_output(look_cmd % "changed").split("\n")
    return filter(lambda _:os.path.splitext(_)[1] in exts, map(lambda _:_[4:], filter(lambda _:_[0] in ("A","U"), filter(len,changed) )))

def main():
    from optparse import OptionParser
    log.debug("pre-commit hook operating\n")
    parser = OptionParser(__doc__)
    parser.add_option("-r", "--revision", help="Test mode. TXN actually refers to a revision.", action="store_true", default=False)
    (opts, (repos, txn_or_rvn)) = parser.parse_args()

    if opts.revision:
        look_cmd = "svnlook %s %s --revision %s " % ("%s", repos, txn_or_rvn)
    else:
        look_cmd = "svnlook %s %s --transaction %s " % ("%s", repos, txn_or_rvn)

    log.info(look_cmd)
    aufiles = au_files(look_cmd, ".cpp .cxx .h .py")
    for name in aufiles: 
        contents = command_output("%s %s" % (look_cmd % "cat", name))
        if "\t" in contents:
            log.warn("tab detected in %s " % name )
            return 255
    return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())


