# TesPa: A simple jit test particle simulator 

This package uses the Boris Method to do a simple test particle simulation by solving the following equations of motion:

$$
    \frac{d\vec{v}}{dt} = \frac{q}{m}(\vec{E} + \vec{v} \times \vec{B})
$$

The initial conditions are loaded from EMSES data as h5 files, using the loadEMSES class. The simulator is the TesPa class.