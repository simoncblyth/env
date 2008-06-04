
from trac.wiki.macros import WikiMacroBase
import re


class HelloWorldMacro(WikiMacroBase):
    """Simple HelloWorld macro.

    Note that the name of the class is meaningful:
     - it must end with "Macro"
     - what comes before "Macro" ends up being the macro name

    The documentation of the class (i.e. what you're reading)
    will become the documentation of the macro, as shown by
    the !MacroList macro (usually used in the WikiMacros page).
    """

    revision = "$Rev$"
    url = "$URL$"
    first_head = re.compile('=\s+([^=]*)=')

    def page_info(self, page):
        from trac.wiki import model

        """ Return tuple of (model.WikiPage, title) """
        page = model.WikiPage(self.env, page)

        title = ''

        if page.exists:
            text = page.text
            ret = self.__class__.first_head.search(text)
            title = ret and ret.group(1) or ''

        return (page, title)

    def expand_macro(self, formatter, name, args):
        """Return some output that will be displayed in the Wiki content.

        `name` is the actual name of the macro (no surprise, here it'll be
        `'HelloWorld'`),
        `args` is the text enclosed in parenthesis at the call of the macro.
          Note that if there are ''no'' parenthesis (like in, e.g.
          [[HelloWorld]]), then `args` is `None`.
        """
        page, title = self.page_info(args)
        return 'Hello World, args = ' + unicode(args) + ' title = ' + unicode(title)
    
    # Note that there's no need to HTML escape the returned data,
    # as the template engine (Genshi) will do it for us.



