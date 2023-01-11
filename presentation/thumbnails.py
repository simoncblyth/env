#!/usr/bin/env python
"""
How to add image links with thumbnails
----------------------------------------

::

   presentation-
   presentation-e     # add meta:thumb annotation string to s5_background_image.txt of images to thumbnail

   cd ~/env/presentation

   ./titles.sh        # creates /tmp/thumb_urls.txt (and several other .txt with urls in them)

   ./thumbnails.sh    # create any new thumbs  

    cd ~/env/simoncblyth.bitbucket.io/images

    vi index.txt    # add figures with thumbs and links to originals 
    make            # update the index


    cd ~/simoncblyth.bitbucket.io

    git s
    git commit / git push etc..
    open https://simoncblyth.bitbucket.io/images/index.html


    cd ~/env/presentation
    ./thumbnails.sh 


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
        #assert url.startswith("http")
        obj = urlparse(url) 
        path = os.path.join("/env/presentation", obj.path )

        img = Image.open(path)
        width, height = img.size
        factor = 4
        if width > 2560: factor = 8 

        print(" url %s width %d height %d factor %d path %s " % (url, width, height, factor, path))

        stub, ext = os.path.splitext(path)
        mkr = "_thumb%d" % factor
        tpath = "%s%s%s" % (stub,mkr, ext) 


        if not os.path.exists(tpath):
            tsize = np.array( img.size, dtype=np.int32 )//factor
            ithumb = tuple(tsize) 
            msg = "creating thumbnail"
            img.thumbnail(ithumb)
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

    print(__doc__)

