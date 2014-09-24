#!/usr/bin/env python
"""
Adapted from the docutils rst2s5.py tool 

A minimal front end to the Docutils Publisher, producing HTML slides using
the S5 template system.

* http://docutils.sourceforge.net/docs/howto/rst-roles.html

TODO:

#. vanilla RST extlinks, similar to sphinx.ext.extlinks
   /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sphinx/ext/extlinks.py


"""


try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

try:
    import IPython as IP
except:
    IP = None



import os, sys, logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import docutils.nodes as nodes
from docutils.core import publish_doctree

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


import s5_video
directives.register_directive('s5_video',s5_video.S5VideoDirective)
add_node( s5_video.s5video, 
    html=(s5_video.visit_s5video_html, s5_video.depart_s5video_html)
)

import s5_background_image
directives.register_directive('s5_background_image',s5_background_image.S5BackgroundImage)
add_node( s5_background_image.s5backgroundimage, 
    html=(s5_background_image.visit_s5backgroundimage_html, 
          s5_background_image.depart_s5backgroundimage_html)
)


from docutils.core import default_description, default_usage
from docutils.core import Publisher 


def collect_resources( doctree ):
    """
    :param doctree: 
    :return urls, paths:

    Collects source urls of images, videos and background images 
    contained in the document by doctree traverse.

    Resolves the urls assuming the DOCBASE envvar points
    to the directory where paths beginning with '/' reside. 
    The DOCBASE will often be the APACHE_HTDOCS folder.   
    """
    urls = []
    imgs = doctree.traverse(nodes.image)
    urls.extend(map(lambda img:img['uri'], imgs))

    vids = doctree.traverse(s5_video.s5video)
    urls.extend(map(lambda vid:vid.arguments[0], vids))

    #bkis = doctree.traverse(s5_background_image.s5backgroundimage)
    bkis = s5_background_image.urls  # kludge
    urls.extend(bkis)

    log.info("collect_resource_urls images: %s s5vids: %s s5bkimg: %s total %s " % (len(imgs),len(vids),len(bkis),len(urls)))
    paths = []
    for i, url in enumerate(urls):
        path = resolve_resource( url , docbase=os.environ['DOCBASE'])
        paths.append(path)
        ok = "OK" if os.path.exists(path) else "??"
        print "%-4d %s %-60s %s " % (i, ok, url, path ) 
    pass
    #log.info("collect_resources end")
    return urls, paths 


def resolve_resource( url, docbase ):
    if url[0] == '/':
        root = docbase
        url = url[1:]
    else:
        root = "."
    pass
    path = os.path.abspath(os.path.join(root, url))
    return path


def main():
    """
    Break out from docutils.core.publish_cmdline to 
    provide access to the doctree.

    Set up & run a `Publisher` for command-line-based file I/O (input and
    output file paths taken automatically from the command line).  Return the
    encoded string output also.

    Parameters: see `publish_programmatically` for the remainder.

    - `argv`: Command-line argument list to use instead of ``sys.argv[1:]``.
    - `usage`: Usage string, output if there's a problem parsing the command
      line.
    - `description`: Program description, output for the "--help" option
      (along with command-line option descriptions).

    """
    reader, reader_name = None, 'standalone'
    parser, parser_name = None, 'restructuredtext'
    writer, writer_name = None, 's5' 
    settings, settings_spec = None, None
    settings_overrides = None
    config_section = None
    enable_exit_status = True
    argv = sys.argv[1:]
    usage = default_usage 
    description = ('Generates S5 (X)HTML slideshow documents from standalone '
                   'reStructuredText sources. ' + default_description )

    pub = Publisher(reader, parser, writer, settings=settings)
    pub.set_components(reader_name, parser_name, writer_name)

    output = pub.publish(
        argv, usage, description, settings_spec, settings_overrides,
        config_section=config_section, enable_exit_status=enable_exit_status)

    urls, paths = collect_resources(pub.document)
    #print "\n".join(paths)

    if not IP is None:
        pass
        #IP.embed()

    return output



if __name__ == '__main__':
    main()
    


