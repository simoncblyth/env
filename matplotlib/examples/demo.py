
import matplotlib
#matplotlib.use('Agg')    # Agg      non-interactive/web server usage
matplotlib.use('TkAgg')  # TkAgg    interactive plot showing etc... 
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1,2,3])

fig.show()


