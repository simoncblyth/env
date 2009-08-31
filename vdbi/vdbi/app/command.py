from vdbi.app import serve_app

def vdbi(argv=None):
    app = serve_app(dbg=True)