from matplotlib import colors as c
import numpy as np
from matplotlib.collections import LineCollection


def plot_phisp(sim,ax,slice,contour=True, alpha=0.5, log=True, levels=10, lintresh=1e-1, linscale=0.5, label_contour=True):
    phi = sim.data.phisp
    if log:
        divnorm=c.SymLogNorm(linthresh=lintresh, linscale=linscale, vmax=np.amax(phi), vmin=-np.amax(phi))#vmin=np.amin(phi), vmax=np.amax(phi))
    else:
        divnorm=c.TwoSlopeNorm(vmin=np.amin(phi), vcenter=0., vmax=np.amax(phi))
    # Plot the potential
    shape = phi[slice].shape 
    extent=[0,shape[0]* sim.data.dx, 0,shape[1]* sim.data.dx]

    im = ax.imshow(phi[slice],origin="lower",aspect="equal",cmap="RdBu", norm=divnorm, extent=extent,)

    if contour:
        cntr = ax.contour(phi[slice],levels=levels,colors="black",alpha=alpha,extent=extent, linestyle="dashed", )
        if label_contour:
            ax.clabel(cntr, fmt="%2.1f", use_clabeltext=True)
    return im


def slice_r(r,dr,ax,):
    i = int(r[ax]/dr)
    if ax == 0:
        return np.s_[i,:,:]
    elif ax == 1:
        return np.s_[:,i,:]
    elif ax == 2:
        return np.s_[:,:,i]


def running_average(data, window_size):
        """
        Compute the running average (moving average) of a 1D array.
        """
        return np.convolve(data, np.ones(window_size), mode='valid') / window_size




def plot_r(x,y,center,data,ax, color, alpha=0.5, x_scale = 1.0, y_scale = 1.0, window_size=100, return_vals=False): 
    X,Y = np.meshgrid(x,y,indexing="xy")
    x0 = center[0]
    y0 = center[1]
    R = np.sqrt((X-x0)**2 + (Y-y0)**2)
    R_flat = R.ravel()     # Flatten R to 1D
    data_flat = data.ravel()  # Flatten data to 1D

    # Step 2: Sort the flattened R array and use the same order for data_flat
    sorted_indices = np.argsort(R_flat)  # Indices that would sort R_flat
    R_sorted = R_flat[sorted_indices]    # Sorted R values (1D)
    data_sorted = data_flat[sorted_indices]

    def running_average(data, window_size):
        """
        Compute the running average (moving average) of a 1D array.
        """
        return np.convolve(data, np.ones(window_size), mode='valid') / window_size

    data_avg = running_average(data_sorted, window_size)
    R_avg = R_sorted[window_size // 2:len(data_avg) + window_size // 2]
    if return_vals:
        return R_avg, data_avg, 
    # Plot the running average and standard deviation
    
    if type(color) == list:
        avg_plot = ax.scatter(R_avg*x_scale,data_avg, color=color[window_size // 2:len(data_avg) + window_size // 2])
        print("Color is list")
    else:
        ax.plot(R_sorted * x_scale,data_sorted, color=color,alpha = alpha)
        avg_plot, = ax.plot(R_avg*x_scale,data_avg, color=color)
    return avg_plot