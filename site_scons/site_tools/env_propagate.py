import os
name = os.path.basename(__file__)
def generate(env, **kw ):
    for k,b in kw.items():
        if b:
            v = os.environ.get(k, None)
            if v:  
                env['ENV'][k] = v
                #print "%s : %s = %s " % ( name , k , v )
            else:
                print "%s : %s = ... not in environ " % ( name , k )





