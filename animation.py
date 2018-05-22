import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation

fig = plt.figure()
plt.axis('equal')
plt.grid()
ax = fig.add_subplot(111)
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)

patch = patches.Rectangle((80, 120), 100, 120, fill=False, color='r')


def init():
    img = mpimg.imread('image2016.jpg')
    ax.imshow(img, extent=[0, 500, 0, 600], aspect='auto', cmap='Greys_r')
    ax.add_patch(patch)
    return patch,


def animate(i):
    patch.set_width(100)
    patch.set_height(160)
    y = 350 - 30 * (i * 0.8 // 500)
    x = i * 0.8 % 500
    patch.set_xy([x, y])
    return patch,


anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=8000,
                               interval=2,
                               blit=True)
plt.show()
