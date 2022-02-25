#!/opt/local/bin/python2.7
"""
titles.py
===========

Invoked by titles.sh for example::

    ./titles.sh opticks_20220115_innovation_in_hep_workshop_hongkong.txt

This parses the .txt source of an s5 presentation collecting background 
image urls from pages matching a titlematch string specified by TITLEMATCH envvar. 
The urls are then written to file /tmp/urls.txt

"""
import logging, os, sys
log = logging.getLogger(__name__)

import docutils.nodes as nodes
from docutils.core import publish_doctree
from docutils.core import default_description, default_usage
from docutils.core import Publisher 


def add_node(node, **kwds):
    """ 
    s5html version of sphinx/application.py:add_node 
    """
    log.debug('adding node: %r', (node, kwds))
    nodes._add_node_class_names([node.__name__])
    for key, val in kwds.iteritems():
        try:
            visit, depart = val 
        except ValueError:
            raise ExtensionError('Value for key %r must be a ' '(visit, depart) function tuple' % key)
        if key == 'html':
            from docutils.writers.s5_html import S5HTMLTranslator as translator
        else:
            continue
        setattr(translator, 'visit_'+node.__name__, visit)
        if depart:
            setattr(translator, 'depart_'+node.__name__, depart)
    pass

from docutils.parsers.rst import directives


import s5_talk
directives.register_directive('s5_talk',s5_talk.S5TalkDirective)
add_node( s5_talk.s5talk, 
    html=(s5_talk.visit_s5talk_html, s5_talk.depart_s5talk_html)
)

import s5_background_image
directives.register_directive('s5_background_image',s5_background_image.S5BackgroundImage)
add_node( s5_background_image.s5backgroundimage, 
    html=(s5_background_image.visit_s5backgroundimage_html, 
          s5_background_image.depart_s5backgroundimage_html)
)




def title_select(titlematch="", prefix=""):

    titles = []
    all_urls = []
    urls = []
    log.debug("-------------- titles  titlematch %s ", titlematch )

    for isect, section in enumerate(doctree.traverse(nodes.section)):
        names = section.attributes['names']
        if len(names) > 0:
            title = names[0]
        else: 
            title = repr(names)
        pass
        log.debug("title:%s" % title)
        titles.append(title)

        div = div_background.Find(title)
        #print(repr(div))

        url = div.urls[0].lstrip().rstrip() if not div is None and len(div.urls) == 1 else None
        if url is None: continue

        assert len(div.meta) == 1
        meta = div.meta[0]

        all_urls.append(url)

        select = (len(titlematch) == 0 or title.find(titlematch) > -1) 

        print(" %d : %s " % (int(select), url) )
        if select:
            urls.append(url)
        pass
    pass
    log.info(" titles:%d titlematch:%s all_urls:%d urls:%d " % (len(titles), titlematch, len(all_urls), len(urls)))

    outpath = "/tmp/urls.txt"
    print("writing %s " % outpath)
    open(outpath, "w").write("\n".join(map(lambda url:"%s%s" % (prefix,url),urls))) 



def thumb_select(prefix=""):
    thumb_div = div_background.FindMeta("thumb")
    print("thumb_div %d " % len(thumb_div))

    urls = []
    for div in thumb_div:
        url = div.urls[0].lstrip().rstrip() if not div is None and len(div.urls) == 1 else None
        if not url in urls:
            urls.append(url)
            print( "thumb_select.url : %s " % url )
        pass  
    pass
    outpath = "/tmp/thumb_urls.txt"
    print("writing %s " % outpath)
    open(outpath, "w").write("\n".join(map(lambda url:"%s%s" % (prefix,url),urls))) 




if __name__ == '__main__':

    #level = logging.INFO
    level = logging.DEBUG
    logging.basicConfig(level=level)

    reader, reader_name = None, 'standalone'
    parser, parser_name = None, 'restructuredtext'
    writer, writer_name = None, 's5' 
    settings, settings_spec = None, None
    settings_overrides = None
    config_section = None
    enable_exit_status = True
    argv = sys.argv[1:]

    log.debug("argv: " + " ".join(argv))

    usage = default_usage 
    description = ('Generates S5 (X)HTML slideshow documents from standalone '
                   'reStructuredText sources. ' + default_description )

    log.debug("Publisher")
    pub = Publisher(reader, parser, writer, settings=settings)

    log.debug("pub.set_components")
    pub.set_components(reader_name, parser_name, writer_name)

    log.debug("[pub.publish")
    output = pub.publish(
        argv, usage, description, settings_spec, settings_overrides,
        config_section=config_section, enable_exit_status=enable_exit_status)
    
    log.debug("]pub.publish")
    doctree = pub.document

    from s5_background_image import urls, divs, div_background
    #for i, url in enumerate(urls): print("%4d : %s " % (i, url))
    #for i, div in enumerate(divs): print("%4d : %s " % (i, repr(div)))

    titlematch = os.environ.get("TITLEMATCH", "")
    prefix = os.environ.get("PREFIX", "")

    title_select(titlematch=titlematch, prefix=prefix)
    thumb_select(prefix=prefix)

