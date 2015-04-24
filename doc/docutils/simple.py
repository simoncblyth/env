"""
::

    python simple.py /tmp/report.rst 

"""
#import IPython

from docutils import writers
from docutils.core import Publisher
import docutils.nodes as nodes


default_usage = '%prog [options] [<source> [<destination>]]'
default_description = ('Reads from <source> (default is stdin) and writes to '
                       '<destination> (default is stdout).  See '
                       '<http://docutils.sf.net/docs/user/config.html> for '
                       'the full reference.')


class Writer(writers.Writer):

    supported = ('pprint', 'pformat', 'pseudoxml')
    """Formats this writer supports."""

    config_section = 'pseudoxml writer'
    config_section_dependencies = ('writers',)

    output = None
    """Final translated form of `document`."""

    def translate(self):
        #self.output = self.document.pformat()
        self.output = "klop\n"

    def supports(self, format):
        """This writer supports all format-specific elements."""
        return True


def main():

    reader=None
    parser=None
    writer=Writer()
    reader_name='standalone'
    parser_name='restructuredtext'
    writer_name= None
    settings = None
    settings_spec = None
    settings_overrides = None
    config_section = None
    enable_exit_status=True

    argv=None
    usage=default_usage
    description=default_description
 
    pub = Publisher(reader, parser, writer, settings=settings)
    pub.set_components(reader_name, parser_name, writer_name)

    output = pub.publish(
        argv, usage, description, settings_spec, settings_overrides,
        config_section=config_section, enable_exit_status=enable_exit_status)


    doc = writer.document
    print dir(doc)
    for section in doc.traverse(nodes.section):
        names = section.attributes['names']
        print names


    #IPython.embed()

    return output



if __name__ == '__main__':
    main()

