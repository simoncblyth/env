import docutils.nodes as nodes





class BaseTranslator(nodes.NodeVisitor):
    def __init__(self, document):
        nodes.NodeVisitor.__init__(self, document)

    def default_visit(self, node):
        self.document.reporter.warning('missing visit_%s' % (node.tagname, ))

 
    def astext(self):
        return self.document.pformat()

    def visit_document(self, node):
        pass
    def depart_document(self, node):
        pass

    def visit_docinfo(self, node):
        pass
    def depart_docinfo(self, node):
        pass

    def visit_field(self, node):
        pass
    def depart_field(self, node):
        pass
    def visit_field_name(self, node):
        pass
    def depart_field_name(self, node):
        pass
    def visit_field_body(self, node):
        pass
    def depart_field_body(self, node):
        pass
    def visit_Text(self, node):
        pass
    def depart_Text(self, node):
        pass
    def visit_paragraph(self, node):
        pass
    def depart_paragraph(self, node):
        pass
    def visit_date(self, node):
        pass
    def depart_date(self, node):
        pass
    def visit_section(self, node):
        pass
    def depart_section(self, node):
        pass
    def visit_title(self, node):
        pass
    def depart_title(self, node):
        pass
    def visit_figure(self, node):
        pass
    def depart_figure(self, node):
        pass
    def visit_image(self, node):
        pass
    def depart_image(self, node):
        pass
    def visit_caption(self, node):
        pass
    def depart_caption(self, node):
        pass
    def visit_bullet_list(self, node):
        pass
    def depart_bullet_list(self, node):
        pass
    def visit_list_item(self, node):
        pass
    def depart_list_item(self, node):
        pass
    def visit_enumerated_list(self, node):
        pass
    def depart_enumerated_list(self, node):
        pass
    def visit_reference(self, node):
        pass
    def depart_reference(self, node):
        pass

    def visit_emphasis(self, node):
        pass
    def depart_emphasis(self, node):
        pass
    def visit_strong(self, node):
        pass
    def depart_strong(self, node):
        pass



