#!/usr/bin/env python

import os
import errno
import sys
import argparse
import math

import ConfigParser
import ast
import json
import collections
from collections import OrderedDict

import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator
import scipy.stats

import uproot

import hepevt

#/////////////////////////////////////////////////////////////
# parse arguments
#/////////////////////////////////////////////////////////////

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to configuration file')
#parser.add_argument('--plot', dest='plot', action='store_true',
#                    help='plot beam and pile-up distributions')
#parser.set_defaults(plot=False)
args = parser.parse_args()

#/////////////////////////////////////////////////////////////
# config parser
#/////////////////////////////////////////////////////////////

# check if configuration file exists
if not os.path.isfile(args.file):
    raise IOError(errno.ENOENT, 'No such file or directory', args.file)
    #print 'No such file or directory:', args.file
    #sys.exit(1)

#config = ConfigParser.RawConfigParser()
#config = ConfigParser.ConfigParser()
config = ConfigParser.SafeConfigParser()
config.read(args.file)

beam_file_path = ast.literal_eval(config.get('config', 'beam_file'))
halo_pileup_file_path = ast.literal_eval(config.get('config', 'halo_pileup_file'))

halo_pileup_on = config.getboolean('config', 'halo_pileup_on')

number_events = config.getint('config', 'events')
number_files = config.getint('config', 'number_files')

output_dir = ast.literal_eval(config.get('config', 'output_dir'))
output_prefix = ast.literal_eval(config.get('config', 'output_prefix'))
output_ext = ast.literal_eval(config.get('config', 'output_ext'))

decoder = json.JSONDecoder(object_pairs_hook=collections.OrderedDict)
pdg_coefficients = decoder.decode(config.get('config', 'pdg_coefficients'))

secondary_beam_energy = config.getfloat('config', 'secondary_beam_energy')

halo_pileup_momentum_low = secondary_beam_energy * 0.57  # GeV
halo_pileup_momentum_high = secondary_beam_energy  # GeV

halo_pileup_pdg_codes = ast.literal_eval(config.get('config', 'halo_pileup_pdg_codes'))

halo_pileup_parameters_list = ast.literal_eval(config.get('config', 'halo_pileup_parameters'))
halo_pileup_bin_counts_txt_file_list = ast.literal_eval(config.get('config', 'halo_pileup_bin_counts_txt'))
halo_pileup_bin_edges_txt_file_list = ast.literal_eval(config.get('config', 'halo_pileup_bin_edges_txt'))

#/////////////////////////////////////////////////////////////
# additional config
#/////////////////////////////////////////////////////////////

tracking_efficiency = 0.56

#/////////////////////////////////////////////////////////////
# physical constants
#/////////////////////////////////////////////////////////////

# particle masses in MeV/c^2
particle_mass = {
    11:   0.5109989461,  # electron
    13:   105.6583745,   # muon
    211:  139.57061,     # charged pion
    321:  493.677,       # charged kaon
    2212: 938.2720813,   # proton
    }

# TPC bounds in the x-coordinate
#tpc_x_min =  0.0  # cm
#tpc_x_max = 47.5  # cm

tpc_anode_x   =   0.0  # cm
tpc_cathode_x =  47.5  # cm

tpc_bottom_y  = -20.0  # cm
tpc_top_y     =  20.0  # cm

tpc_front_z   =   0.0  # cm
tpc_back_z    =  90.0  # cm

# fiducial padding
fiducial_padding_x_low  = 5.0  # cm
fiducial_padding_x_high = 5.0  # cm

fiducial_anode_x   = tpc_anode_x   + 4.0  # cm
fiducial_cathode_x = tpc_cathode_x - 4.0  # cm

# drift velocity
drift_window_ticks = 2528  # ticks
time_per_tick      = 128   # ns
drift_window_time  = drift_window_ticks * time_per_tick
#drift_velocity = (tpc_x_max - tpc_x_min) / drift_window_time
drift_velocity = (tpc_cathode_x - tpc_anode_x) / drift_window_time

#/////////////////////////////////////////////////////////////
# uproot
#/////////////////////////////////////////////////////////////

beam_file = uproot.open(beam_file_path)

beam_tree = beam_file['beam']

number_beam_entries = beam_tree.numentries

beam_arrays = beam_tree.arrays(['x', 'y', 'z', 'angle_xz', 'angle_yz', 'momentum'])
beam_x_array = beam_arrays['x']
beam_y_array = beam_arrays['y']
beam_z_array = beam_arrays['z']
beam_angle_xz_array = beam_arrays['angle_xz']
beam_angle_yz_array = beam_arrays['angle_yz']
beam_momentum_array = beam_arrays['momentum']

halo_pileup_file = None
halo_pileup_xyz_tree = None
halo_pileup_angle_tree = None

number_halo_pileup_xyz_entries = 0
number_halo_pileup_angle_entries = 0

halo_pileup_arrays = None
halo_pileup_x_array = None
halo_pileup_y_array = None
halo_pileup_z_array = None
halo_pileup_angle_xz_array = None
halo_pileup_angle_yz_array = None

if halo_pileup_on:
    halo_pileup_file = uproot.open(halo_pileup_file_path)

    halo_pileup_xyz_tree = halo_pileup_file['halo_pileup_xyz']
    halo_pileup_angle_tree = halo_pileup_file['halo_pileup_angle']

    number_halo_pileup_xyz_entries = halo_pileup_xyz_tree.numentries
    number_halo_pileup_angle_entries = halo_pileup_angle_tree.numentries

    #halo_pileup_xyz_arrays = halo_pileup_xyz_tree.arrays(['x', 'y', 'z'])
    halo_pileup_xyz_arrays = halo_pileup_xyz_tree.arrays(['x', 'y', 'z', 'angle_xz', 'angle_yz'])
    halo_pileup_x_array = halo_pileup_xyz_arrays['x']
    halo_pileup_y_array = halo_pileup_xyz_arrays['y']
    halo_pileup_z_array = halo_pileup_xyz_arrays['z']
    halo_pileup_angle_xz_array = halo_pileup_xyz_arrays['angle_xz']
    halo_pileup_angle_yz_array = halo_pileup_xyz_arrays['angle_yz']

    #halo_pileup_angle_arrays = halo_pileup_angle_tree.arrays(['angle_xz', 'angle_yz'])
    #halo_pileup_angle_xz_array = halo_pileup_angle_arrays['angle_xz']
    #halo_pileup_angle_yz_array = halo_pileup_angle_arrays['angle_yz']

#/////////////////////////////////////////////////////////////
# for plotting
#/////////////////////////////////////////////////////////////

pdg_codes = [ int(pdg_code) for pdg_code in pdg_coefficients.keys() ]

beam_x_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_y_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_z_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_x_proj_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_y_proj_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_z_proj_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_angle_xz_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_angle_yz_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_momentum_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_momentum_x_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_momentum_y_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_momentum_z_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])

halo_pileup_x_dist = []
halo_pileup_y_dist = []
halo_pileup_z_dist = []
halo_pileup_angle_xz_dist = []
halo_pileup_angle_yz_dist = []
halo_pileup_momentum_dist = []
halo_pileup_momentum_x_dist = []
halo_pileup_momentum_y_dist = []
halo_pileup_momentum_z_dist = []
halo_pileup_number_particles_dist = []

halo_pileup_x0_dist = []
halo_pileup_y0_dist = []
halo_pileup_t0_dist = []
halo_pileup_x1_dist = []

#/////////////////////////////////////////////////////////////
# particle beam composition
#/////////////////////////////////////////////////////////////
beam_particle = hepevt.Particle(pdg_coefficients)

#/////////////////////////////////////////////////////////////
# halo pile-up particles
#/////////////////////////////////////////////////////////////
#halo_pileup = hepevt.PileUp(
#    halo_pileup_bin_counts_txt_file_list, halo_pileup_bin_edges_txt_file_list,
#    halo_pileup_parameters_list, number_events, tracking_efficiency)

pk = np.array([
    0.27783235,
    0.27645015,
    0.1854363,
    0.16142433,
    0.08687691,
    0.01197996
    ])

pk = np.array([
    2.54861499e-01, 2.53610941e-01, 1.69887098e-01, 1.49796547e-01,
    7.05333948e-02, 4.57910214e-02, 2.24682110e-03, 4.58568594e-02,
    0.00000000e+00, 0.00000000e+00, 2.53987302e-03, 4.24928167e-03,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 3.38203823e-06, 0.00000000e+00,
    0.00000000e+00, 3.23669004e-05, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.90915279e-04,
    ])

pk = np.array([
    2.55009916e-01, 2.53745277e-01, 1.70129927e-01, 1.48930374e-01,
    7.46507983e-02, 3.41722561e-02, 2.41608140e-02, 2.08388586e-02,
    1.27101191e-02, 3.46261266e-03
    ])

pk = np.array([
    2.53065671e-01, 2.51810674e-01, 1.68832823e-01, 1.47794900e-01,
    7.40816462e-02, 3.39117202e-02, 2.39766072e-02, 2.06799791e-02,
    1.26132147e-02, 3.43621303e-03, 0.00000000e+00, 0.00000000e+00,
    1.66477666e-03, 2.47567231e-03, 1.49596157e-03, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.10900470e-05,
    8.65097147e-04, 1.20997451e-03, 1.11288435e-03, 7.07722368e-04,
    1.73372256e-04
    ])

# for estimating the tracking efficiency
pk = np.array([
    8.22991699e-06, 1.81275985e-01, 1.48416224e-01, 1.21512927e-01,
    9.94863699e-02, 8.14525506e-02, 6.66877081e-02, 5.45992774e-02,
    4.47021075e-02, 3.65989902e-02, 2.99647188e-02, 2.45330368e-02,
    2.00859517e-02, 1.64449863e-02, 1.34640161e-02, 1.10234040e-02,
    9.02519986e-03, 7.38920868e-03, 6.04977239e-03, 4.95313470e-03,
    4.05528370e-03, 3.32018548e-03, 2.71833796e-03, 2.22558688e-03,
    1.82215643e-03, 1.49185550e-03, 1.22142798e-03, 1.00002065e-03,
    8.18747659e-04, 6.70333888e-04, 5.48822969e-04, 4.49338242e-04,
    3.67887038e-04, 3.01200431e-04, 2.46602056e-04, 2.01900687e-04,
    1.65302302e-04, 1.35338078e-04, 1.10805446e-04, 9.07198265e-05,
    7.42751119e-05, 6.08113183e-05, 4.97880964e-05, 4.07630457e-05,
    3.33739591e-05, 2.73242867e-05, 2.23712338e-05, 1.83160171e-05,
    1.49958865e-05, 1.22775934e-05, 1.00520433e-05])

# 2019-04-15

#pk = np.array([
#    0.34531425576879893,   0.2676295025344435,    0.1643234292131439,
#    0.09409932308427549,   0.05625164943190697,   0.0203807603264618,
#    0.02488217294362899,   0.011964345642907603,  9.536740132598531e-07,
#    0.007805218549094328,  0.0042685953336816795, 9.536740132598531e-07,
#    9.536740132598531e-07, 9.536740132598531e-07, 9.536740132598531e-07,
#    9.536740132598531e-07, 0.002158938181341219,  0.0009249036950392808 ])

# 2019-04-16

#pk = np.array([
#    0.32846654178097134, 0.2615971074780511, 0.16355139616894754,
#    0.11099199294808121, 0.05407999743700964, 0.02173093564784534,
#    0.020241161700960852, 0.027712548871429044, 8.165275550142681e-08,
#    6.925188755779743e-10, 0.00037735801650823353, 0.01035508009921432,
#    2.75458465870404e-07, 1.6822566190066723e-07, 1.1215916012607963e-07,
#    2.4003362131264083e-08, 1.1889458861880087e-08, 4.556570559755002e-08,
#    7.386652458585274e-07, 7.719344014661678e-05, 0.000781729499240269,
#    3.778151870331703e-05, 1.6239858846578414e-08, 7.467007229200462e-08,
#    4.412025800260011e-09, 5.167232569602831e-08, 4.350547699871754e-11,
#    1.602807497835812e-09, 9.443612558612813e-11, 1.6866574803486856e-10,
#    2.4253515862326935e-10, 8.075423663100878e-10, 2.490352368766935e-09,
#    8.905973181150273e-10, 9.626976993359904e-10, 2.6002947572933977e-09,
#    1.7100922899970783e-09, 3.050232288970278e-11, 3.922551172763633e-11,
#    3.018352234818167e-11, 1.1745482364489135e-10, 8.235934711997572e-10,
#    2.3500756896055464e-10, 3.60444452063291e-11, 2.2795954013332675e-10,
#    1.9629053937819663e-10, 2.220446049250313e-16, 1.2454076658841018e-10,
#    1.0853035137259326e-10, 5.817157311405197e-10, 4.412986642776673e-11,
#    1.4296896999610453e-11, 1.223757761792399e-10, 3.9004577345735925e-11,
#    3.940470172381083e-11, 1.000447502619295e-10, 1.0736855848847426e-11,
#    4.646766305071992e-11, 7.219169706473849e-12, 3.0919433680054453e-11,
#    1.3506973317589654e-12, 6.314211931091052e-10, 9.361339481372966e-11,
#    3.391453784473697e-11, 7.587097616834626e-12, 6.441513988875158e-13,
#    2.4018509403589405e-10, 1.45530310025066e-10, 1.1749046180398182e-11,
#    5.211275855288022e-12, 2.0848489601377196e-11, 8.030576204021145e-12,
#    2.1143642392473794e-12, 6.578071420904053e-13, 1.8207657603852567e-13,
#    1.8465784457077916e-12, 1.1317558001877615e-11, 2.7755575615628914e-16,
#    1.5489831639570184e-12, 2.8619773217997135e-11, 1.800559701337079e-11,
#    1.4932499681208355e-14, 8.881784197001252e-16, 3.832156814098653e-12,
#    2.711719737646945e-13, 2.4966140266258208e-12, 3.784528246342234e-12,
#    4.747868764809482e-12, 1.6054935159104389e-12, 5.646039191731234e-13,
#    1.122602011349727e-12, 2.362554596402333e-13, 7.216449660063518e-16,
#    2.1549428907974288e-13, 1.2191914144921157e-12, 1.9498846981491624e-12,
#    1.486033518460772e-12, 1.1756151607755783e-12, 1.4551138072249614e-12,
#    1.5666357100485584e-12 ])

# 2019-05-05

pk = np.array([
    2.86577089e-01, 2.41575447e-01, 1.71279961e-01, 1.09202285e-01,
    5.36881321e-02, 6.00613805e-02, 2.09211327e-02, 2.53408069e-09,
    1.89778804e-02, 2.26509755e-02, 1.14720340e-05, 5.22242764e-05,
    1.60275151e-04, 7.04864431e-03, 2.40983688e-08, 1.42963752e-04,
    1.24376629e-05, 1.40360874e-05, 3.46695966e-06, 1.11606163e-04,
    5.02525874e-04, 1.35066435e-03, 1.16807458e-05, 1.36271602e-03,
    7.95897302e-05, 1.81255059e-03, 6.56391633e-05, 3.11197126e-04,
    2.81957314e-04, 7.68494634e-05, 1.79230746e-04, 7.16093891e-05,
    5.92112217e-05, 1.20274746e-05, 1.04426461e-05, 7.43317299e-05,
    8.28749020e-08, 7.29748102e-05, 1.58749418e-04, 2.87861732e-05,
    2.86782715e-05, 3.23893478e-05, 1.70423923e-05, 8.22648812e-05,
    1.97551528e-05, 7.34051234e-06, 1.67387968e-05, 1.50265378e-05,
    7.48559170e-06, 5.18729117e-06, 8.63317612e-06, 1.07442408e-05,
    5.51716754e-06, 8.96880385e-06, 3.41807153e-05, 1.17388656e-05,
    2.15210658e-05, 2.56840939e-05, 3.17821879e-05, 2.81725666e-05,
    1.29718926e-05, 1.65708299e-05, 2.55329860e-05, 2.38196285e-05,
    1.45572031e-05, 4.48700435e-06, 1.49688750e-05, 2.58083013e-05,
    2.27999489e-05, 2.16438252e-05, 1.69687798e-05, 1.68613007e-07,
    7.71938166e-06, 1.43968507e-05, 7.31751858e-06, 7.85294420e-06,
    8.29207868e-06, 3.34602116e-06, 1.92994788e-05, 7.39066697e-06,
    6.17510220e-06, 1.83735406e-05, 2.08292957e-06, 5.70176357e-06,
    1.97590705e-06, 1.11961865e-05, 2.26014544e-06, 9.04177657e-06,
    8.22416354e-06, 8.01660846e-06, 1.51030540e-05, 7.38962278e-07,
    5.28998476e-06, 2.57940778e-05, 4.13666510e-05, 2.69905578e-06,
    2.12392587e-07, 4.18689850e-06, 9.27545746e-08, 1.72701036e-04,
    ])

# 2019-05-06

pk = np.array([
    2.86577089e-01, 2.41575447e-01, 1.71279961e-01, 1.09202285e-01,
    5.36881321e-02, 6.00613805e-02, 2.09211327e-02, 2.53408069e-09,
    1.89778804e-02, 2.26509755e-02, 1.14720340e-05, 5.22242764e-05,
    1.60275151e-04, 7.04864431e-03, 2.40983688e-08, 1.42963752e-04,
    1.24376629e-05, 1.40360874e-05, 3.46695966e-06, 1.11606163e-04,
    5.02525874e-04, 1.35066435e-03, 1.16807458e-05, 1.36271602e-03,
    7.95897302e-05, 1.81255059e-03, 6.56391633e-05, 3.11197126e-04,
    2.81957314e-04, 7.68494634e-05, 1.79230746e-04, 7.16093891e-05,
    5.92112217e-05, 1.20274746e-05, 1.04426461e-05, 7.43317299e-05,
    8.28749020e-08, 7.29748102e-05, 1.58749418e-04, 2.87861732e-05,
    2.86782715e-05, 3.23893478e-05, 1.70423923e-05, 8.22648812e-05,
    1.97551528e-05, 7.34051234e-06, 1.67387968e-05, 1.50265378e-05,
    ])

pk = pk / np.sum(pk)

xk = np.arange(len(pk))

halo_pileup_distribution = scipy.stats.rv_discrete(values=(xk, pk))
number_halo_pileup = halo_pileup_distribution.rvs(size=number_events)

#/////////////////////////////////////////////////////////////
# write to output text file(s)
#/////////////////////////////////////////////////////////////
number_events_per_file = number_events / number_files
digits = int(math.log10(number_files))+1
file_count = 0

print 'Generating %s events...' % number_events
print 'Generating %s file(s) with %s events per file...' % (number_files, number_events_per_file)

f = open(output_dir + output_prefix + str(file_count).zfill(digits) + output_ext, 'w')

for evt_idx in xrange(number_events):

    if evt_idx % number_events_per_file == 0 and evt_idx != 0:
        f.close()
        file_count += 1
        f = open(output_dir + output_prefix + str(file_count).zfill(digits) + output_ext, 'w')
        print 'Writing file %s...' % file_count

    # draw random number for beam entry
    beam_idx = np.random.randint(0, number_beam_entries)

    # initialize number of beam particles and halo pile-up particles
    number_beam = 1
    number_halo_pileup_particles = 0

    # initialize strings for writing to output text file
    beam_str = ''
    halo_pileup_str = ''

    if halo_pileup_on and halo_pileup_xyz_tree is not None and halo_pileup_angle_tree is not None:

        # get number of halo pile-up particles
        number_halo_pileup_particles = number_halo_pileup[evt_idx]

        for pileup_idx in xrange(number_halo_pileup_particles):

            # draw random numbers for halo pile-up entry
            halo_pileup_xyz_idx = np.random.randint(0, number_halo_pileup_xyz_entries)
            halo_pileup_angle_idx = np.random.randint(0, number_halo_pileup_angle_entries)

            # get PDG code of halo_pileup particle
            halo_pileup_pdg_code_ = halo_pileup_pdg_codes[np.random.randint(0, len(halo_pileup_pdg_codes))]

            # get halo pile-up particle position
            halo_pileup_x_ = halo_pileup_x_array[halo_pileup_xyz_idx]
            halo_pileup_y_ = halo_pileup_y_array[halo_pileup_xyz_idx]
            halo_pileup_z_ = halo_pileup_z_array[halo_pileup_xyz_idx]

            # set halo pile-up particle time
            halo_pileup_time_ = 0.0

            halo_pileup_adjust_time_ = False
            halo_pileup_x0_ = halo_pileup_x_
            #halo_pileup_t0_ = halo_pileup_time_
            #halo_pileup_x1_ = halo_pileup_x_

            #if halo_pileup_x_ < fiducial_anode_x or halo_pileup_x_ > fiducial_cathode_x:
            if True:
                halo_pileup_x0_ = np.random.uniform(fiducial_anode_x, fiducial_cathode_x)
                #halo_pileup_t0_ = (halo_pileup_x1_ - halo_pileup_x0_) / drift_velocity
                halo_pileup_time_ = (halo_pileup_x_ - halo_pileup_x0_) / drift_velocity
                halo_pileup_adjust_time_ = True

            # get halo pile-up particle angle
            #halo_pileup_angle_xz_ = halo_pileup_angle_xz_array[halo_pileup_xyz_idx]
            #halo_pileup_angle_yz_ = halo_pileup_angle_yz_array[halo_pileup_xyz_idx]
            #halo_pileup_angle_xz_ = -3
            #halo_pileup_angle_yz_ = 0

            #------------------------------------------------------------------
            # angular magic
            #------------------------------------------------------------------

            #mu = -1.7144085793432629
            #gamma_c = 1.2823824970395936
            #gamma_b = 0.12574821939148592

            #halo_pileup_angle_xz_ = (halo_pileup_angle_xz_ - mu) * (gamma_c - gamma_b) / gamma_c + mu
            #halo_pileup_angle_yz_ = np.random.normal(-4.06138559e-01, 2.292416035600889)

            #a = -4
            #loc = -2 + 0.25
            #scale = 1.5
            #halo_pileup_angle_xz_ = scipy.stats.skewnorm.rvs(a, loc, scale)
            #halo_pileup_angle_yz_ = np.random.normal(-4.06138559e-01, 1.24251617)

            #halo_pileup_angle_xz_ = -3
            #halo_pileup_angle_yz_ = -0.4

            #halo_pileup_angle_xz_ = np.random.uniform(-20, 10)
            #halo_pileup_angle_yz_ = np.random.uniform(-10, 10)

            """
            g = scipy.stats.cauchy.rvs(0, 1)
            angle = np.random.rand() * 2 * np.pi
            x = g * np.cos(angle)
            y = g * np.sin(angle)

            #loc_x = -2.2
            loc_x = -3.0
            #scale_x = 1.2
            scale_x = 1.1

            loc_y = -0.4
            #scale_y = 1.5
            scale_y = 1.3

            x *= scale_x
            x += loc_x

            y *= scale_y
            y += loc_y

            halo_pileup_angle_xz_ = x
            halo_pileup_angle_yz_ = y
            """

            #halo_pileup_angle_xz_, halo_pileup_angle_yz_ = hepevt.angle()
            halo_pileup_angle_xz_, halo_pileup_angle_yz_ = hepevt.angle(
                loc_x=-2.2, scale_x=1.2, loc_y=-0.4, scale_y=1.4, r=45)

            #------------------------------------------------------------------

            #halo_pileup_angle_xz_ = halo_pileup_angle_xz_array[halo_pileup_angle_idx]
            #halo_pileup_angle_yz_ = halo_pileup_angle_yz_array[halo_pileup_angle_idx]

            # time offset

            # start position projection
            halo_pileup_x_proj_, halo_pileup_y_proj_, halo_pileup_z_proj_ = hepevt.projection_at_z(
                -10, halo_pileup_x0_, halo_pileup_y_, halo_pileup_z_,
                halo_pileup_angle_xz_ * np.pi / 180.0, halo_pileup_angle_yz_ * np.pi / 180.0)

            # get halo_pileup particle momentum
            halo_pileup_momentum_ = np.random.uniform(halo_pileup_momentum_low, halo_pileup_momentum_high)

            halo_pileup_momentum_x_, halo_pileup_momentum_y_, halo_pileup_momentum_z_ = hepevt.vector(
                halo_pileup_momentum_, halo_pileup_angle_xz_ * np.pi/180.0, halo_pileup_angle_yz_ * np.pi/180.0)

            # compute halo_pileup particle mass and energy
            halo_pileup_mass_ = particle_mass[np.abs(halo_pileup_pdg_code_)] / 1000.0
            halo_pileup_energy_ = np.sqrt(
                halo_pileup_momentum_ * halo_pileup_momentum_ + halo_pileup_mass_ * halo_pileup_mass_)

            # append to halo pile-up string
            halo_pileup_str += (
                '1' + ' ' +
                str(halo_pileup_pdg_code_)   + ' 0 0 0 0 ' +
                str(halo_pileup_momentum_x_) + ' ' +
                str(halo_pileup_momentum_y_) + ' ' +
                str(halo_pileup_momentum_z_) + ' ' +
                str(halo_pileup_energy_)     + ' ' +
                str(halo_pileup_mass_)       + ' ' +
                #str(halo_pileup_x_)          + ' ' +
                #str(halo_pileup_x0_)         + ' ' +
                #str(halo_pileup_y_)          + ' ' +
                #str(halo_pileup_z_)          + ' ' +
                str(halo_pileup_x_proj_)     + ' ' +
                str(halo_pileup_y_proj_)     + ' ' +
                str(halo_pileup_z_proj_)     + ' ' +
                str(halo_pileup_time_)       + '\n')
                #str(halo_pileup_t0_)         + '\n')

            #halo_pileup_x_proj_, halo_pileup_y_proj_, halo_pileup_z_proj_ = hepevt.projection_at_z(
            #    3, halo_pileup_x_proj_, halo_pileup_y_proj_, halo_pileup_z_proj_,
            #    halo_pileup_angle_xz_ * np.pi / 180.0, halo_pileup_angle_yz_ * np.pi / 180.0)

            #halo_pileup_x_dist.append(halo_pileup_x_proj_)
            #halo_pileup_y_dist.append(halo_pileup_y_proj_)
            #halo_pileup_z_dist.append(halo_pileup_z_proj_)
            halo_pileup_x_dist.append(halo_pileup_x_)
            halo_pileup_y_dist.append(halo_pileup_y_)
            halo_pileup_z_dist.append(halo_pileup_z_)
            halo_pileup_angle_xz_dist.append(halo_pileup_angle_xz_)
            halo_pileup_angle_yz_dist.append(halo_pileup_angle_yz_)
            halo_pileup_momentum_dist.append(halo_pileup_momentum_ * 1000.0)
            halo_pileup_momentum_x_dist.append(halo_pileup_momentum_x_ * 1000.0)
            halo_pileup_momentum_y_dist.append(halo_pileup_momentum_y_ * 1000.0)
            halo_pileup_momentum_z_dist.append(halo_pileup_momentum_z_ * 1000.0)

            if halo_pileup_adjust_time_:
                halo_pileup_x0_dist.append(halo_pileup_x0_)
                halo_pileup_y0_dist.append(halo_pileup_y_)
                halo_pileup_t0_dist.append(halo_pileup_time_ / 1000.0)
                halo_pileup_x1_dist.append(halo_pileup_x_)
                #halo_pileup_t0_dist.append(halo_pileup_t0_ / 1000.0)
                #halo_pileup_x1_dist.append(halo_pileup_x1_)

        halo_pileup_number_particles_dist.append(number_halo_pileup_particles)

    # get total number of particles
    number_particles = number_beam + number_halo_pileup_particles
    f.write(str(evt_idx) + ' ' + str(number_particles) + '\n')

    # get beam particle position
    beam_x_ = beam_x_array[beam_idx]
    beam_y_ = beam_y_array[beam_idx]
    beam_z_ = beam_z_array[beam_idx]

    # get beam particle angle
    beam_angle_xz_ = beam_angle_xz_array[beam_idx]
    beam_angle_yz_ = beam_angle_yz_array[beam_idx]

    # get beam particle momentum
    beam_momentum_ = beam_momentum_array[beam_idx] / 1000.0

    # get PDG code of beam particle
    #beam_pdg_code_ = beam_pdg_codes[np.random.randint(0, len(beam_pdg_codes))]
    beam_pdg_code_ = beam_particle.pdg_from_momentum(beam_momentum_ * 1000.0)

    beam_momentum_x_, beam_momentum_y_, beam_momentum_z_ = hepevt.vector(
        beam_momentum_, beam_angle_xz_ * np.pi/180.0, beam_angle_yz_ * np.pi/180.0)

    # compute beam particle mass and energy
    beam_mass_ = particle_mass[np.abs(beam_pdg_code_)] / 1000.0
    beam_energy_ = np.sqrt(
        beam_momentum_ * beam_momentum_ + beam_mass_ * beam_mass_)

    # set beam particle time
    beam_time_ = 0.0

    # write beam string to output file
    beam_str = (
        '1' + ' ' +
        str(beam_pdg_code_)   + ' 0 0 0 0 ' +
        str(beam_momentum_x_) + ' ' +
        str(beam_momentum_y_) + ' ' +
        str(beam_momentum_z_) + ' ' +
        str(beam_energy_)     + ' ' +
        str(beam_mass_)       + ' ' +
        str(beam_x_)          + ' ' +
        str(beam_y_)          + ' ' +
        str(beam_z_)          + ' ' +
        str(beam_time_))

    f.write(beam_str + '\n')

    beam_x_proj_, beam_y_proj_, beam_z_proj_ = hepevt.projection_at_z(
        0, beam_x_, beam_y_, beam_z_, beam_angle_xz_ * np.pi / 180.0, beam_angle_yz_ * np.pi / 180.0)

    beam_x_dist[beam_pdg_code_].append(beam_x_)
    beam_y_dist[beam_pdg_code_].append(beam_y_)
    beam_z_dist[beam_pdg_code_].append(beam_z_)
    beam_x_proj_dist[beam_pdg_code_].append(beam_x_proj_)
    beam_y_proj_dist[beam_pdg_code_].append(beam_y_proj_)
    beam_z_proj_dist[beam_pdg_code_].append(beam_z_proj_)
    beam_angle_xz_dist[beam_pdg_code_].append(beam_angle_xz_)
    beam_angle_yz_dist[beam_pdg_code_].append(beam_angle_yz_)
    beam_momentum_dist[beam_pdg_code_].append(beam_momentum_ * 1000.0)
    beam_momentum_x_dist[beam_pdg_code_].append(beam_momentum_x_ * 1000.0)
    beam_momentum_y_dist[beam_pdg_code_].append(beam_momentum_y_ * 1000.0)
    beam_momentum_z_dist[beam_pdg_code_].append(beam_momentum_z_ * 1000.0)

    # if halo pile-up TTree objects are present
    if halo_pileup_on and halo_pileup_xyz_tree is not None and halo_pileup_angle_tree is not None and not not halo_pileup_str:
        f.write(halo_pileup_str)

beam_plotter = hepevt.Plotter('beam', config)

beam_plotter.plot_position(beam_x_dist, beam_y_dist, beam_z_dist)
#beam_plotter.plot_position(beam_x_proj_dist, beam_y_proj_dist, beam_z_proj_dist)
beam_plotter.plot_momentum(beam_momentum_dist,
                           beam_momentum_x_dist,
                           beam_momentum_y_dist,
                           beam_momentum_z_dist,
                           beam_angle_xz_dist,
                           beam_angle_yz_dist)

if halo_pileup_on:
    halo_pileup_plotter = hepevt.Plotter('halo_pileup', config)
    halo_pileup_plotter.plot_position(halo_pileup_x_dist, halo_pileup_y_dist, halo_pileup_z_dist)
    halo_pileup_plotter.plot_momentum(halo_pileup_momentum_dist,
                                      halo_pileup_momentum_x_dist,
                                      halo_pileup_momentum_y_dist,
                                      halo_pileup_momentum_z_dist,
                                      halo_pileup_angle_xz_dist,
                                      halo_pileup_angle_yz_dist)
    halo_pileup_plotter.plot_number_particles(halo_pileup_number_particles_dist)
    halo_pileup_plotter.plot_time_offset(halo_pileup_x0_dist,
                                         halo_pileup_y0_dist,
                                         halo_pileup_t0_dist,
                                         halo_pileup_x1_dist)

