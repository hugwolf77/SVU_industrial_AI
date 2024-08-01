import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

m = 1 # kg 추의 무게
k = 1 # N/m  스프링의 탄성 강도
d = 0.1 # Unit of d

t = np.linspace(0,40, 501)
w_d = np.sqrt((4*m*k - d**2)/(4*m**2))
x = np.exp(-d/(2*m)*t) * np.cos(w_d*t)

fig, axis = plt.subplots(1, 2)
animated_mess, = axis[0].plot([],[], 'o', markersize=20,  color='red')
animated_spring, = axis[0].plot([],[], color='blue') # ',' is used because aixs.plot returns an array
axis[0].set_xlim([-2, 2])
axis[0].set_ylim([-2, 2])
axis[0].set_title("Title-1")
axis[0].grid()

animated_disp, = axis[1].plot([],[], color='red')
axis[1].set_xlim([min(t), max(t)])
axis[1].set_ylim([-2, 2])
axis[1].set_title("Title-2")
axis[1].grid()


def plot_update(frame):
    animated_mess.set_data([x[frame]], [0]) # Updating the data across [frame]
    animated_spring.set_data([-2,x[frame]],[0,0])
    animated_spring.set_linewidth(int(abs(x[frame]-2)*2))
    animated_disp.set_data(t[:frame], x[:frame])
    return animated_mess, animated_spring, animated_disp

animation = FuncAnimation(
    					fig=fig,
						func=plot_update,
						frames=len(t),
						interval=25,
                        blit=True,
						repeat=True,
						)

plt.show()


