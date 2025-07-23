import numpy as np

def TranslatePoints(
        points:np.ndarray, values:np.ndarray, lattice_vecs:np.ndarray
)->tuple[np.ndarray, np.ndarray]:
    npts = len(points)
    # translate points along each lattice direction to ensure grid space is filled
    trans_points = np.zeros((27*npts,3))
    trans_values = np.zeros((27*npts,1))
    for i in range(3):
        even_ind = 9*i
        odd_ind = 9*i+1 
        # translate points along z axis
        layer_pts = points + (i-1)*lattice_vecs[2,:]
        trans_points[even_ind*npts:(even_ind+1)*npts,:] = layer_pts
        # translate along +x and -x
        trans_points[odd_ind*npts:(odd_ind+1)*npts,:] = layer_pts + lattice_vecs[0,:]
        trans_points[(even_ind+2)*npts:(even_ind+3)*npts,:] = layer_pts - lattice_vecs[0,:]
        # along +y and -y
        trans_points[(odd_ind+2)*npts:(odd_ind+3)*npts,:] = layer_pts + lattice_vecs[1,:]
        trans_points[(even_ind+4)*npts:(even_ind+5)*npts,:] = layer_pts - lattice_vecs[1,:]
        # along +x+y and +x-y
        trans_points[(odd_ind+4)*npts:(odd_ind+5)*npts,:] = layer_pts + lattice_vecs[0,:] + lattice_vecs[1,:]
        trans_points[(even_ind+6)*npts:(even_ind+7)*npts,:] = layer_pts - lattice_vecs[0,:] + lattice_vecs[1,:]
        # along -x+y and -x-y
        trans_points[(odd_ind+6)*npts:(odd_ind+7)*npts,:] = layer_pts + lattice_vecs[0,:] - lattice_vecs[1,:]
        trans_points[(even_ind+8)*npts:(even_ind+9)*npts,:] = layer_pts - lattice_vecs[0,:] - lattice_vecs[1,:]
        # repeat translations for values array
        trans_values[even_ind*npts:(even_ind+1)*npts,:] = values
        trans_values[odd_ind*npts:(odd_ind+1)*npts,:] = values
        trans_values[(even_ind+2)*npts:(even_ind+3)*npts,:] = values
        trans_values[(odd_ind+2)*npts:(odd_ind+3)*npts,:] = values
        trans_values[(even_ind+4)*npts:(even_ind+5)*npts,:] = values
        trans_values[(odd_ind+4)*npts:(odd_ind+5)*npts,:] = values
        trans_values[(even_ind+6)*npts:(even_ind+7)*npts,:] = values
        trans_values[(odd_ind+6)*npts:(odd_ind+7)*npts,:] = values
        trans_values[(even_ind+8)*npts:(even_ind+9)*npts,:] = values
    return trans_points, trans_values