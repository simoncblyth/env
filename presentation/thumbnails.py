#!/usr/bin/env python
"""
thumbnails.py
===============

::

   ./titles.sh        # update the /tmp/thumb_urls.txt according to meta:thumb additions to s5_background_image.txt

   ./thumbnails.sh   # create any new thumbs  


    cd ~/env/simoncblyth.bitbucket.io/images

    vi index.txt    # add figures with thumbs and links to originals 
    make            # update the index


    cd ~/simoncblyth.bitbucket.io

    git s
    git commit / git push etc..
    open https://simoncblyth.bitbucket.io/images/index.html


"""

import os, logging, textwrap
from urllib.parse import urlparse
import PIL
from PIL import Image
import numpy as np

i = np.array([1920, 1080])
i4 = i//4 


class FigRST(dict):
    tmpl = textwrap.dedent("""

    .. figure:: %(thumb_url)s

        `%(short_caption)s. <%(image_url)s>`_
    """) 

    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)

    def __str__(self):
        return self.tmpl % self 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    path = "/tmp/thumb_urls.txt"
    urls = open(path).read().splitlines()
    log.info(" path %s urls %d " % (path, len(urls)))

    imgs = []
    figs = []

    for i, url in enumerate(urls):
        assert url.startswith("http")
        obj = urlparse(url) 
        path = obj.path

        stub, ext = os.path.splitext(path)
        mkr = "_thumb4"   
        tpath = "%s%s%s" % (stub,mkr, ext) 

        img = Image.open(path)
        width, height = img.size

        if not os.path.exists(tpath):
            tsize = np.array( img.size, dtype=np.int32 )//4 
            i4 = tuple(tsize) 
            msg = "creating thumbnail"
            img.thumbnail(i4)
            img.save(tpath)
        else:
            msg = "thumbnail exists already"
        pass

        thumb = Image.open(tpath)

        print(" %10s : %10s : %80s : %80s  : %s " % (str(img.size), str(thumb.size), path, tpath, msg ))
        fig = FigRST(short_caption="Placeholder Caption %d" % i, thumb_url=tpath, image_url=path)
        figs.append(fig)
        pass
    pass

    print("\n\n".join(map(str, figs)))


