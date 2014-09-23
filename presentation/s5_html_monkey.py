"""
Latest Safari Web Debugger has a "Disable Cache" control
that probably means this monkey patch to forcibly 
kill the cache is no longer needed.

"""
import time
import docutils.writers.s5_html as s5

class MonkeyWriter(s5.Writer):
    pass

class MonkeyS5HTMLTranslator(s5.S5HTMLTranslator):
    s5_stylesheet_template = """\
<!-- configuration parameters -->
<meta name="defaultView" content="%(view_mode)s" />
<meta name="controlVis" content="%(control_visibility)s" />
<!-- style sheet links -->
<script src="%(path)s/slides.js#Q#" type="text/javascript"></script>
<link rel="stylesheet" href="%(path)s/slides.css#Q#"
      type="text/css" media="projection" id="slideProj" />
<link rel="stylesheet" href="%(path)s/outline.css#Q#"
      type="text/css" media="screen" id="outlineStyle" />
<link rel="stylesheet" href="%(path)s/print.css#Q#"
      type="text/css" media="print" id="slidePrint" />
<link rel="stylesheet" href="%(path)s/opera.css#Q#"
      type="text/css" media="projection" id="operaFix" />\n""".replace("#Q#","?monkeykillcache=%s" % time.time() )
    pass




