"""
    This is now for debugging only ... the real convertions
    use converter.latex2sphinx 


"""
import os, sys
from converter import DocParser, Tokenizer, RestWriter 
from converter import restwriter

# NOT "." as latex running happens one level up, but sphinx conversion only done inside database folder
source = ".."  
class IncludeRewrite:
    """
          rewriting a dict access, 
          
    """
    def get(self, a, b=None):
        if os.path.exists(os.path.join(source, a + '.tex')):
            return a + '.rst'
        return a
restwriter.includes_mapping = IncludeRewrite()

def _convert_file(inf, outf, doraise=True, splitchap=False,
                 toctree=None, deflang=None, labelprefix=''):
    p = DocParser(Tokenizer(inf.read()).tokenize(), inf)
    r = RestWriter(outf, splitchap, toctree, deflang, labelprefix)
    r.write_document(p.parse())
    #outf.close()
    p.finish()  # print warnings about unrecognized commands

if __name__ == '__main__':
    _convert_file( sys.stdin , sys.stdout )

