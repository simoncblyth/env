
from docutils import nodes
from docutils.parsers.rst import Directive, directives

from sphinx import addnodes
from sphinx.util import parselinenos

class StockChart(Directive):
    """
    Avoid duplicating stockchart javascript and div snippet for every plot 
    using this directive.  
    
    Usage::

        .. stockchart:: /data/scm_backup_monitor_C.json container_C

    """
    has_content = True
    required_arguments = 2
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'linenos': directives.flag,
    }

    tmpl = r"""
    <script type="text/javascript" >
    $(function() {
        $.getJSON('%(jsonurl)s', function(options) {
                window.chart = new Highcharts.StockChart(options);
        });
    });
    </script>
    <div id="%(id)s" style="height: 500px; min-width: 500px"></div>
    """

    def run(self):
        jsonurl = self.arguments[0]
        id = self.arguments[1]
        html = self.tmpl % locals()
        raw = nodes.raw('',html, format = 'html')
        raw.document = self.state.document
        return [raw]


def setup(app):
    app.add_directive('stockchart', StockChart)




