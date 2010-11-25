
import simplejson as json
from datetime import datetime

s = dict(name="hello",start=datetime.utcnow(), end=datetime.now())


def handler(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    else:
        raise TypeError, 'Object of type %s with value of %s is not JSON serializable' % (type(obj), repr(obj))


js = json.dumps(s, default=handler)


print js

