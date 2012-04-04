import logging
log = logging.getLogger(__name__)

class Value(dict):
    qwns = dict(number=0.,unit=1.,limit=0.,cl=90.)
class Err(dict):
    qwns = dict(errname="s",type="s",symm=0.,plus=0.,minus=0.) 

class Quote(object):
    """
        <rez:value>
                <rez:number>1.5</rez:number>
                <rez:unit>0.00001</rez:unit>
                <rez:signif>4.9</rez:signif>
                <rez:ltype>-</rez:ltype>
                <rez:limit/>
                <rez:cl/>
            </rez:value>
            <rez:errors>
                <rez:err>
                    <rez:errname>stat</rez:errname>
                    <rez:type>absolute</rez:type>
                    <rez:symm/>
                    <rez:plus>0.5</rez:plus>
                    <rez:minus>0.4</rez:minus>
                </rez:err>
                <rez:err>
                    <rez:errname>syst</rez:errname>
                    <rez:type>absolute</rez:type>
                    <rez:symm>0.1</rez:symm>
                    <rez:plus/>
                    <rez:minus/>
                </rez:err>
            </rez:errors>
            <rez:xerrors>
                <rez:xerr>
                    <rez:errname>Ds BR</rez:errname>
                    <rez:type>absolute</rez:type>
                    <rez:symm>0.2</rez:symm>
                    <rez:plus/>
                    <rez:minus/>
                </rez:xerr>
            </rez:xerrors>


    """


    def __init__(self, rdr):
	self.count = dict(value=0,err=0,xerr=0)
	self.value = Value()
	self.err = [] 
        self.xerr = [] 
        self.read(rdr)

    def __repr__(self):
        return "\n".join( [self.__class__.__name__, repr(self.count), repr(self.value)] + map(repr,self.err) + map(repr,self.xerr) )

    def read(self, rdr ):
        """
        :param rdr: XmlEventReader instance
	"""
	posn = None
        curr = None
        while rdr.hasNext():
	    typ = rdr.next()	
	    if typ == rdr.StartElement:
	        if not rdr.isEmptyElement():    
		    curr = rdr.getLocalName()
                    if curr in ("value","err","xerr"):
                        posn = curr    
	    elif typ == rdr.Characters:
 		self.parse(posn, curr, None if rdr.isWhiteSpace() else rdr.getValue() )		
	    elif typ == rdr.EndElement:
	        curr = None
	    else:	     
		log.info("Q2V ?? typ %s  " % ( typ ))		 


    def parse(self, posn, name, chars):
	"""
	:param posn: one of "value", "err" or "xerr"
	:param name: element local name
	:param chars: string content of the element 

	"""
        if not posn:return
	log.debug(" %-10s : %-30s : %s " % (posn, name, chars))

        def atof(a, nan):
            try:
		return float(a)
	    except TypeError:
            	return nan    
	    except ValueError:
		return nan    

	if posn == name:  
            self.count[name] += 1		
	
	if posn == "value":

            val = self.value
            if name in val.qwns:
                val[name] = atof(chars, val.qwns[name])		

        elif posn in ("err","xerr"):

	    if posn == name:  
	        getattr(self,posn).append(Err())
            err = getattr(self,posn)[-1] 		

            if name in err.qwns: 
		t = err.qwns[name]
                err[name] = chars if t == 's' else atof(chars,t)

        else:
	    pass	



