#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adapted from the docutils rst2s5.py tool 

A minimal front end to the Docutils Publisher, producing HTML slides using
the S5 template system.

* http://docutils.sourceforge.net/docs/howto/rst-roles.html

TODO:

#. vanilla RST extlinks, similar to sphinx.ext.extlinks
   /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sphinx/ext/extlinks.py


"""

from __future__ import print_function
try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

try:
    import IPython as IP
except:
    IP = None



import os, sys, logging, codecs, re, textwrap
log = logging.getLogger(__name__)


#FMT  = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
FMT = '{%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FMT)

import docutils.nodes as nodes
from docutils.core import publish_doctree

def add_node(node, **kwds):
    """ 
    s5html version of sphinx/application.py:add_node 
    """
    log.debug('adding node: %r', (node, kwds))
    nodes._add_node_class_names([node.__name__])
    for key, val in kwds.items():
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


from docutils.core import default_description, default_usage
from docutils.core import Publisher 


def make_talk(title, tk=None):
    if tk is None:
        tk = s5_talk.s5talk()
        tk.content = []
    else: 
        pass
    pass
    tk.title = title
    return tk 


def collect_titles( doctree ):
    """
    Slide titles
    """
    titles = []
    comments = []
    talks = []


    maintitle = str(doctree.traverse(nodes.title)[0][0])
    titles.append(maintitle)
    comments.append([])

    pt = re.compile("<[^>]*>")
    utitle,_ = pt.subn("", maintitle)
    utitle = "%0.2d %s " % (0, utitle)
    print(utitle)


    tks = doctree.traverse(s5_talk.s5talk)
    if len(tks) > 0: 
        print(" tks %s " % tks)
        talks.append( make_talk(utitle, tks[0]) ) 
        with_talk = True 
    else:
        with_talk = False 
    pass


    for isect, section in enumerate(doctree.traverse(nodes.section)):
        names = section.attributes['names']
        if len(names) > 0:
            title = names[0]
        else: 
            title = repr(names)
        pass
        print(title)
        titles.append(title)

        cmms = section.traverse(nodes.comment) 
        cmms = filter(lambda cmm:cmm.astext().startswith("comment"), cmms )
        comments.append(cmms)

        if with_talk:
            utitle = "%0.2d %s " % (isect+1, title)

            tks = section.traverse(s5_talk.s5talk) 
            #print(tks)
            if len(tks) == 0:
               tk = make_talk(utitle) 
            else:
               assert len(tks) == 1 , (len(tks), tks) 
               tk = tks[0]
               tk.title = utitle
            pass
            talks.append(tk)
        pass

    if not IP is None:
        pass
        #IP.embed()

    return titles, comments, talks







def collect_resources( doctree, dump=False ):
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
        if dump:
            print("%-4d %s %-60s %s " % (i, ok, url, path )) 
        pass
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



def dump(titles, comments):
    print("\n\n")
    assert len(titles) == len(comments)
    for i in range(len(titles)):
        title = titles[i]  
        cmms = comments[i]
        print("%0.2d : %2d : %s " % (i, len(cmms), title ))
        print("=" * ( 15 + len(title) ))
        print("\n")
        if 1: continue

        for cmm in cmms:
            print("\n") 
            txt = cmm.astext()
            assert txt.startswith("comment")
            lines = txt.split("\n")

            print("\n".join(map(lambda line:"    %s" % line, lines[1:])))
            print("\n") 
        pass
        print("\n") 
    pass



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

    path = argv[-1]
    name = os.path.basename(path) 
    stem, ext = os.path.splitext(name)
    assert ext == ".html", (ext, name, path) 

    log.info("argv: " + " ".join(argv))

    usage = default_usage 
    description = ('Generates S5 (X)HTML slideshow documents from standalone '
                   'reStructuredText sources. ' + default_description )

    pub = Publisher(reader, parser, writer, settings=settings)
    pub.set_components(reader_name, parser_name, writer_name)

    output = pub.publish(
        argv, usage, description, settings_spec, settings_overrides,
        config_section=config_section, enable_exit_status=enable_exit_status)


    log.info("list of urls and resolved files from collect_resources") 
    urls, paths = collect_resources(pub.document)
    #print "\n".join(paths)
    log.info("list of titles from collect_titles") 
    titles, comments, talks = collect_titles(pub.document)

    assert len(titles) == len(comments)
    #dump(titles, comments)


    #out = sys.stdout 

    stem = stem.replace("_","-")
    tpath = os.path.join("/tmp", "%s.rst" % stem )
    log.info(tpath)
    out = codecs.open(tpath, encoding='utf-8', mode='w')   

    hdr = textwrap.dedent("""
    :title: %s
    :date: Oct 2019

    Invisible Title
    ===================

    """ % stem )
     
    print(hdr, file=out)
    twiddle = u"≈"

    for tk in talks:
        print(tk.title)

        if tk.title.find(twiddle) > -1:
            utitle = tk.title.replace(twiddle,"")
        else:
            utitle = tk.title
        pass
        print(utitle,file=out) 

        print("-" * len(utitle), file=out)


        print("\n", file=out)
        print("\n".join(tk.content), file=out) 
        print("\n\n", file=out)
    pass
    return pub.document



if __name__ == '__main__':
    doc = main()
    


