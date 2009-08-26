

from rum.policy import Policy, Denial
class DbiPolicy(Policy):pass
#
#    def has_permission(self, obj, action, attr=None, user=None):
#         return True


def anyone(policy, obj, action,  attr, user):
    print "dbipolicy callable called ... "
    return True

DbiPolicy.register(anyone)


