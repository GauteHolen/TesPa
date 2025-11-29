import testparticle.tespa.tespa as tespa
import numpy as np
import matplotlib.pyplot as plt

# Create a TesPa object
sim = tespa.TesPa()
# Tell the object where to load the EMSES files
sim.load_emses("../data/sub_vx_fp_magz_small", 0.0025)

# If B is constant it must we set manually as EMSES data does not write it
sim.set_B(90e-6,dir="z")

# Print some info about the simulation
print(sim)

m = 9.10938356e-31 # Particle mass
q = -1.60217662e-19 # Particle charge
r0 = np.array([0.3, 0.2, 0.3]) # Initial position in m
v0 = np.array([1.0e6, 0.0, 0.0]) # Initial speed in m/s
dt0 = 1e-9 # Inital time step for the adaptive time step method



m = 9.10938356e-31 # Particle mass
q = -1.60217662e-19 # Particle charge
r0 = np.array([0.3, 0.2, 0.3]) # Initial position in m
v0 = np.array([1.0e6, 0.0, 0.0]) # Initial speed in m/s
dt0 = 1e-9 # Inital time step for the adaptive time step method
sim.run_simple(r0,v0,m,q,dt0=dt0,Nt = 10000)
print(r0)

plt.figure()
print(sim.r.shape)
print(sim.r[:,0])
plt.plot(sim.r[0,:],sim.r[1,:])
print(sim.r[:,0])
#plt.xlim(0,sim.data.x[-1])
#plt.ylim(0,sim.data.y[-1])

plt.figure()
idx = 1000
plt.plot(sim.t[:idx],sim.r[0,:idx])
plt.show()