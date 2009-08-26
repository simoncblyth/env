from vdbi import debug_here, JOIN_POSTFIX_
from vdbi import is_vld_attr
from rum.policy import Policy, Denial

from inspect import isclass


class DbiPolicy(Policy):pass

def anyone(policy, obj, action,  attr, user):
    #debug_here()
    #print "policy %s obj %s action %s attr %s user %s " % ( policy , obj, action, attr, user )
    
    if isclass(obj):
        name = obj.__name__.lower()
    else:
        name = obj.__class__.__name__.lower()
        
    #debug_here()
        
    if action == "index":
        if attr == None:
            return True
            #return Denial("No access to obj %s is allowed  " % obj )      ## this causes a 403 Forbidden, with the denial message   
        elif name.endswith(JOIN_POSTFIX_) and is_vld_attr(attr):   
            return Denial("Skipping vld attributes for the joined pay:vld object index page ")
    return True


DbiPolicy.register(anyone)


