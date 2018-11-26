import numpy as np

def rotate_xz(x, y, z, angle_xz):

    x_ = x * np.cos(angle_xz) + z * np.sin(angle_xz)
    y_ = y
    z_ = - x * np.sin(angle_xz) + z * np.cos(angle_xz)

    return np.array([ x_, y_, z_ ])

def rotate_yz(x, y, z, theta_yz):

    x_ = x
    y_ = y * np.cos(angle_yz) + z * np.sin(angle_yz)
    z_ = - y * np.sin(angle_yz) + z * np.cos(angle_yz)

    return np.array([ x_, y_, z_ ])

def rotate(x, y, z, angle_xz, angle_yz):

    x_ = x * np.cos(angle_xz) + z * np.sin(angle_xz)
    y_ = y
    z_ = - x * np.sin(angle_xz) + z * np.cos(angle_xz)

    x__ = x_
    y__ = y_ * np.cos(angle_yz) + z_ * np.sin(angle_yz)
    z__ = - y_ * np.sin(angle_yz) + z_ * np.cos(angle_yz)

    return np.array([ x__, y__, z__ ])

def projection_at_z(z, x0, y0, z0, angle_xz, angle_yz):

    x1, y1, z1 = rotate(
        0, 0, 1, angle_xz * np.pi/180.0, angle_yz * np.pi/180.0)

    slope = (z - z0) / z1
    x = x0 + slope * x1
    y = y0 + slope * y1

    return x, y, z