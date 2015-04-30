#!/usr/bin/env python
"""
BitBucket Issue Import/Export Data Format : a json file with attachments directory.
JSON format documented  

* https://confluence.atlassian.com/pages/viewpage.action?pageId=330796872

When using the issue tracker import/export feature to move issues from one
Bitbucket repo to another, the issue IDs remain the same. However, the comment
IDs and therefore the permlinks, change. Comment IDs change because, in
contrast to issue IDs, they are not local to the repo. 

Certain objects, such as comments, rely on foreign keys.  During an import,
Bitbucket silently uses NULL to replace any foreign keys that it cannot resolve
(for example, a username that no longer exists on Bitbucket).



Problems
-----------

#. initially the trac2bitbucket export got all times wrong due to 
   a trac version specific factor of 1 million in timestamps 

   Manually fixed this in working copy.

Questions
-----------

#. what to do about usernames ?

   * at least maqm, lint need to be operative



"""
import json, sys, os, logging
log = logging.getLogger(__name__)

try:
   import IPython
except ImportError:
   IPython = None



def readjson(path):
    with open(path,"rb") as fp:
        js = json.load(fp)
    return js


class BBI(object):
    def __init__(self, js):
        assert len(js) == 8
        self.js = js
        self.check_attachments()

    def check_attachments(self):
        for d in self.js['attachments']:
            print d 

def main():
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1]
    basename = os.path.basename(path)
    name, ext = os.path.splitext(basename)
    assert ext == '.json', ext 
    log.info("basename %s name %s " % (basename,name) )
    js = readjson(path)
    bbi = BBI(js)

    if not IPython is None:
        IPython.embed()  


if __name__ == '__main__':
    main()

