import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
p0, = ax.plot([], [], 'o')
p1, = ax.plot([], [], 'o')

def animate(i):
    x = i/10
    y = i/10
    p0.set_data(x, y)
    p1.set_data(-x, y)
    return p0,p1

ax.set_xlim(-10,10)
ax.set_ylim(0,10)

ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
plt.show()


