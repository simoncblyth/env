#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

class BulletSpacer(object):
    TOKEN = "* "

    @classmethod
    def is_bulletline(cls, line):
        pos = line.find(cls.TOKEN)
        return pos > -1 and len(line[:pos].strip()) == 0   # token preceeded by whitespace

    @classmethod
    def is_blankline(cls, line):
        return len(line.strip()) == 0   

    @classmethod
    def spaced_out(cls, text):
        lines = text.split("\n")
        assert not cls.is_bulletline(lines[0]) 
        s_lines = []
        s_lines.append(lines[0])
        for i in range(1, len(lines)):
            prev = lines[i-1]
            line = lines[i]
            next_ = lines[i+1] if i+1 < len(lines) else None

            p_bullet = cls.is_bulletline(prev)
            p_blank = cls.is_blankline(prev)
            bullet = cls.is_bulletline(line)
            n_bullet = next_ is not None and cls.is_bulletline(next_) 
            n_blank = next_ is not None and cls.is_blankline(next_)
 
            if bullet and not p_blank and not p_bullet:   ## start of bullets without a spacer
                s_lines.append("")
            pass

            s_lines.append(line)

            if bullet and not n_bullet and not n_blank:  ## end of bullets without a spacer
                s_lines.append("")
            pass
        pass
        return "\n".join(s_lines)

    @classmethod
    def applyfix(cls, text, fix_ = lambda line:line):
        lines = text.split("\n")
        f_lines = lines[:]
        for i in range(len(lines)):
            fixline = fix_(lines[i])
            if fixline != lines[i]:
               f_lines[i] = fixline
               log.info("applyfix changed line %s " % i )  
               log.info("bef [%s] " % lines[i] )  
               log.info("aft [%s] " % f_lines[i] )  
            pass
        pass
        return "\n".join(f_lines) 
 
        


def prep(txt):
    return "\n".join(map(lambda _:_[4:], txt.split("\n")[1:-1]))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    text = prep(r"""
    Line before the bullets
     * first  
     * second

     * third
     * fourth
    Line after the bullets
    """) 


    x_text = prep(r"""
    Line before the bullets

     * first  
     * second

     * third
     * fourth

    Line after the bullets
    """) 

    s_text = BulletSpacer.spaced_out(text)

    print text
    print s_text 

    assert s_text == x_text 





