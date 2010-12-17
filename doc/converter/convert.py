import sys
import codecs
from converter import DocParser, Tokenizer, RestWriter 


def _convert_file(inf, outf, doraise=True, splitchap=False,
                 toctree=None, deflang=None, labelprefix=''):
    p = DocParser(Tokenizer(inf.read()).tokenize(), inf)
    r = RestWriter(outf, splitchap, toctree, deflang, labelprefix)
    r.write_document(p.parse())
    outf.close()
    p.finish()  # print warnings about unrecognized commands

if __name__ == '__main__':
    _convert_file( sys.stdin , sys.stdout )

