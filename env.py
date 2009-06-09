import os 
tmpl = """os.environ["%s"] = "%s" """
print "\n".join( [  tmpl % (k,v) for k,v in os.environ.items() ])

