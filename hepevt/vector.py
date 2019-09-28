import numpy as np

tpc_anode_x   =   0.0  # cm
tpc_cathode_x =  47.5  # cm

tpc_bottom_y  = -20.0  # cm
tpc_top_y     =  20.0  # cm

tpc_front_z   =   0.0  # cm
tpc_back_z    =  90.0  # cm

def vector(r, angle_xz, angle_yz):

    z = np.sqrt(r*r) / np.sqrt(1 + np.tan(angle_xz) * np.tan(angle_xz) + np.tan(angle_yz) * np.tan(angle_yz))
    x = z * np.tan(angle_xz)
    y = z * np.tan(angle_yz)

    return np.array([ x, y, z ])

def projection_at_x(x, x0, y0, z0, angle_xz, angle_yz):

    x1, y1, z1 = vector(1, angle_xz, angle_yz)

    slope = (x - x0) / x1
    y = y0 + slope * y1
    z = z0 + slope * z1

    return x, y, z

def projection_at_y(y, x0, y0, z0, angle_xz, angle_yz):

    x1, y1, z1 = vector(1, angle_xz, angle_yz)

    slope = (y - y0) / y1
    x = x0 + slope * x1
    z = z0 + slope * z1

    return x, y, z

def projection_at_z(z, x0, y0, z0, angle_xz, angle_yz):

    x1, y1, z1 = vector(1, angle_xz, angle_yz)

    slope = (z - z0) / z1
    x = x0 + slope * x1
    y = y0 + slope * y1

    return x, y, z

def inside_tpc(x, y, z):
    #if (x >= -4.0          and
    #    x <= 56.0          and
    if (x >= tpc_anode_x   and
        x <= tpc_cathode_x and
        y >= tpc_bottom_y  and
        y <= tpc_top_y     and
        z >= tpc_front_z   and
        z <= tpc_back_z):
        return True
    return False

def inside_tpc_drift(x, y, z):
    if (x >= -4.0          and
        x <= 56.0          and
        y >= tpc_bottom_y  and
        y <= tpc_top_y     and
        z >= tpc_front_z   and
        z <= tpc_back_z):
        return True
    return False

def intersect_tpc_cathode(x, y, z, angle_xz, angle_yz):

    proj_tpc_cathode_x_plane = projection_at_x(tpc_cathode_x, x, y, z, angle_xz, angle_yz)

    if inside_tpc(*proj_tpc_cathode_x_plane):
        return True

    return False

def intersect_tpc(x, y, z, angle_xz, angle_yz):

    proj_tpc_anode_x_plane = projection_at_x(tpc_anode_x, x, y, z, angle_xz, angle_yz)
    proj_tpc_cathode_x_plane = projection_at_x(tpc_cathode_x, x, y, z, angle_xz, angle_yz)

    proj_tpc_bottom_y_plane = projection_at_y(tpc_bottom_y, x, y, z, angle_xz, angle_yz)
    proj_tpc_top_y_plane = projection_at_y(tpc_top_y, x, y, z, angle_xz, angle_yz)

    proj_tpc_front_z_plane = projection_at_z(tpc_front_z, x, y, z, angle_xz, angle_yz)
    proj_tpc_back_z_plane = projection_at_z(tpc_back_z, x, y, z, angle_xz, angle_yz)

    if (inside_tpc(*proj_tpc_anode_x_plane)   or
        inside_tpc(*proj_tpc_cathode_x_plane) or
        inside_tpc(*proj_tpc_bottom_y_plane)  or
        inside_tpc(*proj_tpc_top_y_plane)     or
        inside_tpc(*proj_tpc_front_z_plane)   or
        inside_tpc(*proj_tpc_back_z_plane)):
        return True

    return False

def tpc_intersection_point(x, y, z, angle_xz, angle_yz):

    proj_tpc_anode_x_plane = projection_at_x(tpc_anode_x, x, y, z, angle_xz, angle_yz)
    proj_tpc_cathode_x_plane = projection_at_x(tpc_cathode_x, x, y, z, angle_xz, angle_yz)

    proj_tpc_bottom_y_plane = projection_at_y(tpc_bottom_y, x, y, z, angle_xz, angle_yz)
    proj_tpc_top_y_plane = projection_at_y(tpc_top_y, x, y, z, angle_xz, angle_yz)

    proj_tpc_front_z_plane = projection_at_z(tpc_front_z, x, y, z, angle_xz, angle_yz)
    proj_tpc_back_z_plane = projection_at_z(tpc_back_z, x, y, z, angle_xz, angle_yz)

    intersection_point = False
    z_buffer = 100.0

    #print x, y, z, z_buffer

    if inside_tpc(*proj_tpc_anode_x_plane) and z_buffer > proj_tpc_anode_x_plane[2]:
        intersection_point = proj_tpc_anode_x_plane
        #print '  ', intersection_point, z_buffer
        #z_buffer = proj_tpc_anode_x_plane[2]
        z_buffer = intersection_point[2]

    if inside_tpc(*proj_tpc_cathode_x_plane) and z_buffer > proj_tpc_cathode_x_plane[2]:
        intersection_point = proj_tpc_cathode_x_plane
        #print '  ', intersection_point, z_buffer
        #z_buffer = proj_tpc_cathode_x_plane[2]
        z_buffer = intersection_point[2]

    if inside_tpc(*proj_tpc_bottom_y_plane) and z_buffer > proj_tpc_bottom_y_plane[2]:
        intersection_point = proj_tpc_bottom_y_plane
        #print '  ', intersection_point, z_buffer
        #z_buffer = proj_tpc_bottom_y_plane[2]
        z_buffer = intersection_point[2]

    if inside_tpc(*proj_tpc_top_y_plane) and z_buffer > proj_tpc_top_y_plane[2]:
        intersection_point = proj_tpc_top_y_plane
        #print '  ', intersection_point, z_buffer
        #z_buffer = proj_tpc_top_y_plane[2]
        z_buffer = intersection_point[2]

    if inside_tpc(*proj_tpc_front_z_plane) and z_buffer > proj_tpc_front_z_plane[2]:
        intersection_point = proj_tpc_front_z_plane
        #print '  ', intersection_point, z_buffer
        #z_buffer = proj_tpc_front_z_plane[2]
        z_buffer = intersection_point[2]

    if inside_tpc(*proj_tpc_back_z_plane) and z_buffer > proj_tpc_back_z_plane[2]:
        intersection_point = proj_tpc_back_z_plane
        #print '  ', intersection_point, z_buffer
        #z_buffer = proj_tpc_back_z_plane[2]
        z_buffer = intersection_point[2]

    #print '    ', intersection_point

    return intersection_point

