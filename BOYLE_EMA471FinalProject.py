import numpy as np 
from numpy import sqrt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter as pw

#Constants
G = 6.6743E-11
R = 384.4E6 # Radius of the Earth to the Sun (m)
pi = np.pi # pi constant
mom = 7.3476E22 # Mass of the moon (kg)
moe = 5.9724E24 # Mass of the earth (kg)
mos = 10000 # Mass of the satellite, negligible compared to the other bodies (kg)
v_moon = 1023 # Tangential velocity of moon (m/s)

mtot = (moe + mom)
mu = (mom / mtot)

omega = sqrt(G*(mom + moe) / R**3)
gam = 3*sqrt(3)*(moe-mom) / (4*(mom + moe))

T = (2*pi) / omega # Orbital period of the moon around the Earth

# Defining System of Four First Order ODEs
def dSdt(S, t):
    x, y, vx, vy = S
    return [vx,     
            vy, 
            2*omega*vy + (3/4)*omega**2*x + gam*omega**2*y,
            -2*omega*vx + (9/4)*omega**2*y + gam*omega**2*x]

# Initial Conditions, 4 IC's for 4 ODEs
x1 = 0 
v0x = -v_moon*np.cos(60)
y1 = 0
v0y = v_moon*np.sin(60)

# Initial Condition array
S_0 = np.array([x1, y1, v0x, v0y])

#Time array
t = np.linspace(0, 10*T, int(T))

# Solve the system of differential equations
sol = odeint(dSdt, S_0, t)

x = sol[:, 0]
y = sol[:, 1]
vx = sol[:, 2]
vy = sol[:, 3]

#Velocity
vmag = np.sqrt(vx**2 + vy**2)

figr, axes = plt.subplots()
axes.plot(x,y)
axes.set_xlabel("X Position Relative to L4")
axes.set_ylabel("Y Position Relative to L4")
plt.title("Position of the Satellite Mass over 10 Orbital Periods")
plt.show()

plt.figure(figsize = (8,6))
plt.plot(t, vmag, label = 'Speed of Satellite Mass')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Speed of Satellite Mass around Lagrange Point L4 vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Define the Contour Plot
fig, ax = plt.subplots()
plt.title("Gravitational Potential in the Earth-Moon System")
abs = np.linspace(-1.7, 1.7, 100)
ord = np.linspace(-1.7, 1.7, 100)
X, Y = np.meshgrid(abs, ord)
plt.scatter(0.5-mu, sqrt(3)/2, color='navy', label='L4')
u = -((1-mu)/np.sqrt((X+mu)**2+Y**2)) - (mu / np.sqrt((X-(1-mu))**2 + Y**2)) - (1/2)*(X**2+Y**2)
contour = plt.contour(X, Y, u, levels = 1000)
plt.colorbar(contour)
plt.legend()
plt.grid()
plt.show()

# Animate the x, y positions of the object
def animate(i, xdata=[], ydata=[]):
    plt.cla()
    xdata.append(x[i*1000])
    ydata.append(y[i*1000])
    plt.plot(xdata, ydata)
    plt.xlabel("X Location Relative to L4")
    plt.ylabel("Y Location Relative to L4")
    plt.title("Path of the Satellite Mass over 10 Orbital Periods")
    
fig = plt.figure()
ani = FuncAnimation(fig, animate, interval=1, save_count=1000)
plt.show()
