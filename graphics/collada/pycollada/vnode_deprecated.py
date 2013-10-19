



def parse_collada( path , usecache=False ):
    """
    :param path: to collada file

    #. `collada.Collada` parses the .dae 
    #. a list of bound geometry is obtained from `dae.scene.objects`
    #. `VNode.recurse` traverses the raw pycollada node tree, creating 
       an easier to navigate VNode heirarchy which has one VNode per bound geometry  
    #. cross reference between the bound geometry list and the VNode tree

    """
    import collada 
    path = os.path.expandvars(path)
    log.info("pycollada parse %s " % path )
    dae = collada.Collada(path)
    log.info("pycollada parse completed ")
    boundgeom = list(dae.scene.objects('geometry'))
    top = dae.scene.nodes[0]
    log.info("pycollada binding completed, found %s  " % len(boundgeom))

    log.info("create VNode heirarchy ")
    if usecache and os.path.exists(VNode.pkpath):
        VNode.load()
    else:
        VNode.recurse(top)
        VNode.summary()
        if usecache:
            VNode.save()
    
    VNode.indexlink( boundgeom )
    #VNode.walk()
    return boundgeom


class VNode(object):

    @classmethod
    def save(cls):
        log.info("saving to %s " % cls.pkpath )
        pickle.dump( cls.registry, open( cls.pkpath, "wb" ) ) 

    @classmethod
    def load(cls):
        log.info("loading from %s " % cls.pkpath )
        cls.registry = pickle.load( open( cls.pkpath, "rb" ) ) 
        for v in cls.registry:
            if v.index == 0:
                cls.root = v
                break


