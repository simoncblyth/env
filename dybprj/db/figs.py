"""
    http://matplotlib.sourceforge.net/faq/howto_faq.html#matplotlib-in-a-web-application-server
"""
import matplotlib 
#matplotlib.use('Agg')    # Agg      non-interactive/web server usage
matplotlib.use('TkAgg')  # TkAgg    interactive plot showing etc... 
import matplotlib.pyplot as plt

def demo():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    return fig

def field_hist( dbname, tabname, field ):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    return fig


if __name__ == '__main__':
    fig = demo()
    fig.show()




