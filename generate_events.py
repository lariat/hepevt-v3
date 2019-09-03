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
halo_pileup_side_entering_on = config.getboolean('config', 'halo_pileup_side_entering_on')

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

fiducial_anode_x   = tpc_anode_x   + 5.0  # cm
fiducial_cathode_x = tpc_cathode_x - 5.0  # cm

# drift velocity
drift_window_ticks = 2528  # ticks
time_per_tick      = 128   # ns
drift_window_time  = drift_window_ticks * time_per_tick  # ns
#drift_velocity = (tpc_x_max - tpc_x_min) / drift_window_time
drift_velocity = (tpc_cathode_x - tpc_anode_x) / drift_window_time  # cm / ns
drift_velocity = 0.146964e-3  # cm / ns

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
halo_pileup_tree = None

number_halo_pileup_entries = 0

halo_pileup_arrays = None
halo_pileup_x_array = None
halo_pileup_y_array = None
halo_pileup_z_array = None
halo_pileup_angle_xz_array = None
halo_pileup_angle_yz_array = None
halo_pileup_front_entering_array = None
halo_pileup_side_entering_array = None

if halo_pileup_on:

    halo_pileup_file = uproot.open(halo_pileup_file_path)
    halo_pileup_tree = halo_pileup_file['data_halo_pileup']

    number_halo_pileup_entries = halo_pileup_tree.numentries

    halo_pileup_arrays = halo_pileup_tree.arrays(
        [ 'upstream_x', 'upstream_y', 'upstream_z', 'angle_xz', 'angle_yz',
         'front_entering', 'side_entering' ])

    halo_pileup_x_array = halo_pileup_arrays['upstream_x']
    halo_pileup_y_array = halo_pileup_arrays['upstream_y']
    halo_pileup_z_array = halo_pileup_arrays['upstream_z']
    halo_pileup_angle_xz_array = halo_pileup_arrays['angle_xz']
    halo_pileup_angle_yz_array = halo_pileup_arrays['angle_yz']
    halo_pileup_front_entering_array = halo_pileup_arrays['front_entering']
    halo_pileup_side_entering_array = halo_pileup_arrays['side_entering']

    if not halo_pileup_side_entering_on:

        flags = halo_pileup_side_entering_array == 0
        number_halo_pileup_entries = np.sum(flags)

        halo_pileup_x_array = halo_pileup_arrays['upstream_x'][flags]
        halo_pileup_y_array = halo_pileup_arrays['upstream_y'][flags]
        halo_pileup_z_array = halo_pileup_arrays['upstream_z'][flags]
        halo_pileup_angle_xz_array = halo_pileup_arrays['angle_xz'][flags]
        halo_pileup_angle_yz_array = halo_pileup_arrays['angle_yz'][flags]
        halo_pileup_front_entering_array = halo_pileup_arrays['front_entering'][flags]
        halo_pileup_side_entering_array = halo_pileup_arrays['side_entering'][flags]

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

# 2019-05-06; -100A pimue

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

# 2019-06-24; +100A proton

pk = np.array([
    4.01025220e-01, 2.63688626e-01, 1.51533424e-01, 8.00866727e-02,
    4.32796988e-02, 2.56867787e-02, 1.00265462e-02, 1.13623950e-02,
    4.05980740e-08, 4.01538817e-03, 5.62905133e-03, 3.44579962e-04,
    2.53798831e-04, 1.26553189e-04, 1.57437981e-04, 1.82312695e-04,
    6.77663626e-05, 4.75468600e-04, 1.46920149e-06, 2.10849386e-04,
    2.70553264e-04, 1.26111400e-04, 4.27320249e-04, 3.86369946e-04,
    2.40700048e-04, 6.73793997e-05, 9.53799500e-08, 6.68215698e-07,
    3.36556546e-05, 2.20074760e-05, 1.66683624e-05, 1.31864733e-05,
    1.08041957e-05, 9.10870121e-06, 7.85985311e-06, 6.91226857e-06,
    6.17460753e-06, 5.58751361e-06, 5.11122533e-06, 4.71837397e-06,
    4.38965873e-06, 4.11117069e-06, 3.87269085e-06, 3.66658003e-06,
    3.48703818e-06, 3.32959959e-06, 3.19078237e-06, 3.06784073e-06,
    ])

# 2019-08-08; +100A proton, includes side-entering pile-up

pk = np.array([
    3.38917163e-01, 2.40248541e-01, 1.51872983e-01, 8.98058674e-02,
    5.21999332e-02, 4.66090421e-02, 1.10192037e-02, 2.31868792e-02,
    4.95690052e-07, 2.49627193e-02, 4.74866402e-05, 4.88178435e-05,
    5.39194181e-05, 8.12510555e-08, 2.68591234e-12, 1.27572264e-02,
    2.27671345e-03, 1.22049360e-04, 8.62782255e-05, 1.08385043e-05,
    1.53724871e-06, 9.13453360e-08, 1.15995359e-05, 8.03038987e-06,
    7.46981534e-06, 1.18276170e-05, 1.56022472e-06, 1.39495323e-05,
    5.24985108e-06, 2.40902817e-05, 3.00117260e-05, 5.19079165e-08,
    4.48191231e-06, 5.43996287e-05, 9.27059280e-05, 9.14377069e-05,
    6.12314274e-05, 1.62188751e-04, 6.74168538e-05, 8.30936531e-05,
    6.34621911e-05, 3.48418414e-05, 3.95540077e-04, 3.66949924e-04,
    3.02295409e-04, 2.25197577e-04, 3.51455594e-04, 1.80825173e-04,
    #
    3.46931271e-04, 1.83148646e-04, 1.54016984e-04, 2.14684533e-04,
    3.80683923e-04, 1.01497061e-07, 4.20783181e-04, 8.35455375e-06,
    1.05848380e-05, 2.16936443e-04, 1.58770321e-04, 5.20257103e-06,
    6.46280286e-05, 5.87215506e-05, 1.27823935e-05, 3.29285049e-06,
    3.81016601e-06, 2.01031926e-06, 5.10139468e-07, 2.10108507e-07,
    3.48565605e-07, 1.25543835e-07, 1.52197730e-07, 1.70156395e-08,
    9.43200084e-08, 4.83410240e-09, 8.56855173e-08, 5.85163268e-08,
    1.23665695e-08, 2.97108073e-08, 4.06190080e-09, 2.37356903e-08,
    2.27423065e-08, 2.00402383e-09, 1.08827290e-08, 2.14466335e-09,
    3.32774036e-09, 3.17425162e-09, 4.55950396e-09, 4.19767801e-09,
    2.42151892e-09, 2.41389667e-09, 3.13720087e-09, 4.60715691e-10,
    2.46167441e-09, 1.96331212e-09, 1.06837951e-09, 1.12062659e-09,
    4.63121773e-10, 2.12734509e-10, 1.83601212e-10, 6.51590058e-04,
    ])


# 2019-08-18; +100A proton, includes side-entering pile-up

pk = np.array([
    3.83831243e-01, 2.58661152e-01, 1.54225045e-01, 8.27985523e-02,
    4.74123510e-02, 3.28948981e-02, 5.30816356e-03, 2.10918311e-02,
    2.77446306e-08, 6.17573922e-07, 3.43808947e-05, 1.02859277e-02,
    1.32770352e-03, 7.07929238e-08, 3.59742815e-07, 2.54084035e-07,
    1.47249572e-07, 5.58702261e-08, 2.21798779e-08, 2.75955653e-08,
    5.16341208e-08, 5.12147007e-08, 3.87113937e-07, 3.63684870e-09,
    7.82092516e-06, 4.52039739e-05, 5.67874030e-05, 1.49986118e-04,
    7.71459296e-05, 2.85232009e-04, 2.69862573e-08, 7.41248269e-04,
    5.42241743e-04, 1.32631248e-04, 1.05261689e-17, 1.71852474e-09,
    1.33869775e-08, 1.21062070e-06, 4.75466926e-09, 6.81317182e-09,
    6.47508022e-07, 5.47228133e-07, 3.80978693e-07, 1.57199055e-08,
    3.47456168e-07, 2.92784271e-07, 2.46088828e-07, 2.06831953e-07,
    #
    # 1.74117656e-07, 1.52114008e-07, 1.31657229e-07, 1.12046976e-07,
    # 9.37969775e-08, 7.77005580e-08, 6.70469722e-08, 5.81543000e-08,
    # 5.07090100e-08, 4.44559405e-08, 3.91875055e-08, 3.47347151e-08,
    # 3.09598178e-08, 2.77503260e-08, 2.50141936e-08, 2.26759336e-08,
    # 2.06735000e-08, 1.89557827e-08, 1.74805949e-08, 1.62130559e-08,
    # 1.57375324e-08, 1.59331310e-08, 1.57656691e-08, 1.52697927e-08,
    # 1.44754025e-08, 1.34084110e-08, 1.20914711e-08, 1.09427482e-08,
    # 1.06982337e-08, 1.05225804e-08, 1.04132623e-08, 1.03691128e-08,
    # 1.03903327e-08, 1.04785512e-08, 1.06369477e-08, 1.08704418e-08,
    # 1.11859648e-08, 1.33334819e-08, 1.59156253e-08, 1.82939564e-08,
    # 2.03367455e-08, 2.18625518e-08, 2.26188101e-08, 2.22511560e-08,
    # 2.02622004e-08, 5.36455252e-08, 1.23921294e-08, 2.89342765e-08,
    # 2.64378783e-08, 5.04919789e-16, 2.33428958e-08, 1.95535010e-04,
    ])

# 2019-08-18; +100A proton, includes side-entering pile-up

pk = np.array([
    3.51578702e-01, 2.46128482e-01, 1.52548315e-01, 9.10370912e-02,
    5.70283851e-02, 3.90093690e-02, 1.43957274e-03, 4.14995485e-02,
    2.20166303e-10, 1.89646617e-09, 8.08362074e-09, 1.15319514e-13,
    7.95698667e-03, 7.82031095e-03, 4.00440541e-05, 1.36850984e-05,
    2.00139972e-08, 5.41339000e-05, 1.62568592e-05, 6.32330408e-08,
    8.13931754e-08, 2.21679951e-03, 7.64130274e-07, 1.43283914e-06,
    3.04689816e-06, 2.22055504e-06, 3.25106416e-06, 8.64940989e-07,
    1.89183808e-07, 8.36890212e-07, 2.54635255e-08, 6.03597290e-09,
    1.80948714e-07, 1.05547052e-08, 4.29345276e-07, 2.98537731e-07,
    3.42375377e-10, 6.98531259e-07, 2.91486780e-07, 3.64414234e-07,
    8.22101289e-07, 2.88815172e-06, 5.10044040e-06, 2.73796352e-06,
    9.45327569e-06, 1.17507523e-05, 9.90715578e-06, 4.04555166e-05,
    #
    # 6.41154985e-05, 1.02424845e-04, 6.37238673e-05, 5.49113697e-05,
    # 1.36418432e-04, 3.59682564e-04, 2.33621747e-05, 5.64695123e-04,
    # 3.23753031e-07, 5.24723662e-06, 1.51350044e-06, 2.86270666e-20,
    # 2.76395147e-08, 3.20603944e-08, 4.14968564e-08, 3.91659387e-08,
    # 1.16486007e-08, 5.18332387e-09, 3.77714156e-09, 2.61276145e-09,
    # 5.24431462e-10, 7.80513988e-11, 1.91687979e-11, 1.95261213e-20,
    # 3.13013239e-11, 2.54811894e-10, 3.66749750e-10, 4.54386424e-10,
    # 8.01161063e-10, 9.73790438e-11, 2.64435665e-10, 2.03989929e-10,
    # 9.71166218e-10, 1.40808341e-10, 1.08231762e-10, 4.83349761e-10,
    # 3.78084864e-10, 6.05076692e-10, 1.05447944e-10, 4.65477188e-10,
    # 4.64071196e-10, 1.06889693e-10, 4.87185793e-10, 1.12595827e-10,
    # 1.84054401e-10, 1.13216054e-09, 1.37249498e-10, 5.99240835e-10,
    # 5.27187758e-10, 5.61858695e-10, 9.53989846e-10, 1.36223634e-04,
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

    if halo_pileup_on and halo_pileup_tree is not None:

        # get number of halo pile-up particles
        number_halo_pileup_particles = number_halo_pileup[evt_idx]

        for pileup_idx in xrange(number_halo_pileup_particles):

            # draw random numbers for halo pile-up entry
            halo_pileup_idx = np.random.randint(0, number_halo_pileup_entries)

            # front- or side-entering halo pile-up particle
            halo_pileup_front_entering = halo_pileup_front_entering_array[halo_pileup_idx]
            halo_pileup_side_entering = halo_pileup_side_entering_array[halo_pileup_idx]

            # get PDG code of halo_pileup particle
            halo_pileup_pdg_code_ = halo_pileup_pdg_codes[np.random.randint(0, len(halo_pileup_pdg_codes))]

            # get halo pile-up particle position
            halo_pileup_x_ = halo_pileup_x_array[halo_pileup_idx]
            halo_pileup_y_ = halo_pileup_y_array[halo_pileup_idx]
            halo_pileup_z_ = halo_pileup_z_array[halo_pileup_idx]

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

            if halo_pileup_side_entering:
                halo_pileup_x0_ = tpc_cathode_x
                halo_pileup_x_ = np.random.uniform(-2, 56)
                #halo_pileup_z_ = np.random.uniform(5, 85)
                #halo_pileup_z_ = 100
                halo_pileup_time_ = (halo_pileup_x_ - tpc_cathode_x) / drift_velocity
                halo_pileup_adjust_time_ = True

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

            # time offset

            # start position projection
            halo_pileup_x_proj_, halo_pileup_y_proj_, halo_pileup_z_proj_ = hepevt.projection_at_z(
                -40, halo_pileup_x0_, halo_pileup_y_, halo_pileup_z_,
                halo_pileup_angle_xz_ * np.pi / 180.0, halo_pileup_angle_yz_ * np.pi / 180.0)

            if halo_pileup_side_entering:

                halo_pileup_x_proj_ = tpc_cathode_x
                #halo_pileup_x_proj_ = halo_pileup_x0_
                halo_pileup_y_proj_ = halo_pileup_y_
                halo_pileup_z_proj_ = halo_pileup_z_ 

                halo_pileup_x_proj_, halo_pileup_y_proj_, halo_pileup_z_proj_ = hepevt.projection_at_z(
                    -20, tpc_cathode_x, halo_pileup_y_, halo_pileup_z_,
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

            if halo_pileup_adjust_time_ and halo_pileup_side_entering:
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

            #if halo_pileup_adjust_time_:
            if halo_pileup_adjust_time_ and halo_pileup_side_entering:
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
    if halo_pileup_on and halo_pileup_tree is not None and not not halo_pileup_str:
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

