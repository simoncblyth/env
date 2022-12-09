"""
animate_point_path.py
=======================


      3----------2
      |          |
  Z   |          |
      |          |
      |          |     
      0----------1
          X

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()


post0 = np.array([ 
            [0,0,0,0],
            [1,0,0,10],   # +X  
            [1,0,1,20],   # +X+Z
            [0,0,1,30],
            [0,0,0,40]], dtype=np.float32 )


post1 = post0.copy() 
#post1[:,3] += 5   ## changes the lengths, causing assert

post1[:,0] += 0.05 


def interpolated_post_array(post, tstep=0.1):
    t = post[:,3]
    tt = np.arange(0,t[-1],tstep) 
    it = np.digitize(tt, t)  ## 1,1,1,........,2,2,........,5,5,5,5
    def point(idx):
        """
        TODO: find more vectorized way to pre-compute the interpolated post array 
        """
        t = tt[idx]
        i = it[idx]
        a = post[i-1]
        b = post[i]
        f = (t - a[3])/(b[3]-a[3])
        c = a*(1-f) + f*b 
        return c 
    pass
    ipost = np.zeros( (len(tt),4), dtype=np.float32 )
    idxs = np.arange(len(tt))
    for idx in idxs: ipost[idx] = point(idx)
    return ipost, idxs

ipost0, idxs0 = interpolated_post_array(post0)
ipost1, idxs1 = interpolated_post_array(post1)

assert np.all( idxs0 == idxs1 )
idxs = idxs0


p0, = ax.plot([], [], 'o')
p1, = ax.plot([], [], 'o')

def animate(idx):
    c0 = ipost0[idx]
    c1 = ipost1[idx]
    p0.set_data(c0[0],c0[2])
    p1.set_data(c1[0],c1[2])
    return p0,p1

ax.set_xlim(-1.1,1.1)
ax.set_ylim(-0.1,1.1)

ani = animation.FuncAnimation(fig, animate, frames=idxs, interval=50, blit=True, repeat=False)
plt.show()


