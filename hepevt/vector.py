import numpy as np

def vector(r, angle_xz, angle_yz):

    z = np.sqrt(r) / np.sqrt(1 + np.tan(angle_xz) * np.tan(angle_xz) + np.tan(angle_yz) * np.tan(angle_yz))
    x = z * np.tan(angle_xz)
    y = z * np.tan(angle_yz)

    return np.array([ x, y, z ])

def projection_at_z(z, x0, y0, z0, angle_xz, angle_yz):

    x1, y1, z1 = vector(1, angle_xz, angle_yz)

    slope = (z - z0) / z1
    x = x0 + slope * x1
    y = y0 + slope * y1

    return x, y, z
