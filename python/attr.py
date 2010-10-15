

class Ex(object):
    def __init__(self):
        self.msg = None
    def __getattribute__(self, attr):
        print "intercept", attr
        return super(Ex, self).__getattribute__(attr)
    def get_msg(self):
        return self.msg
    def set_msg(self, msg):
        self.msg = msg
    m = property( get_msg, set_msg  )


if __name__=='__main__':
    e = Ex()
    e.m = "jo"    
    


