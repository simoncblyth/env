"""
https://stackoverflow.com/questions/46849712/how-can-i-animate-a-set-of-points-with-matplotlib

"""

import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def neighbors(point):
    x, y = point
    for i, j in itertools.product(range(-1, 2), repeat=2):
        if any((i, j)):
            yield (x + i, y + j)

def advance(board):
    newstate = set()
    recalc = board | set(itertools.chain(*map(neighbors, board)))

    for point in recalc:
        count = sum((neigh in board)
                for neigh in neighbors(point))
        if count == 3 or (count == 2 and point in board):
            newstate.add(point)

    return newstate

glider = set([(0, 0), (1, 0), (2, 0), (0, 1), (1, 2)])

fig, ax = plt.subplots()

x, y = zip(*glider)
mat, = ax.plot(x, y, 'o')

def animate(i):
    global glider
    glider = advance(glider)
    x, y = zip(*glider)
    mat.set_data(x, y)
    return mat,

ax.axis([-15,5,-15,5])
ani = animation.FuncAnimation(fig, animate, interval=50)
plt.show()
