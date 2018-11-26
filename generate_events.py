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

beam_file = ast.literal_eval(config.get('config', 'beam_file'))
halo_pileup_file = ast.literal_eval(config.get('config', 'halo_pileup_file'))

halo_pileup_on = config.getboolean('config', 'halo_pileup_on')

number_events = config.getint('config', 'events')
number_files = config.getint('config', 'number_files')

output_dir = ast.literal_eval(config.get('config', 'output_dir'))
output_prefix = ast.literal_eval(config.get('config', 'output_prefix'))
output_ext = ast.literal_eval(config.get('config', 'output_ext'))

decoder = json.JSONDecoder(object_pairs_hook=collections.OrderedDict)
pdg_coefficients = decoder.decode(config.get('config', 'pdg_coefficients'))

secondary_beam_energy = config.getfloat('config', 'secondary_beam_energy')

halo_pileup_momentum_low = secondary_beam_energy / 2.0  # GeV
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

#/////////////////////////////////////////////////////////////
# ROOT
#/////////////////////////////////////////////////////////////

import ROOT

beam_tfile = ROOT.TFile.Open(beam_file, 'read')
halo_pileup_tfile = None

# check if TTree object exists
if not beam_tfile.GetListOfKeys().Contains('beam'):
    print 'Cannot find TObject: beam'
    sys.exit(1)

beam_ttree = beam_tfile.Get('beam')
halo_pileup_ttree = None

number_beam_entries = beam_ttree.GetEntries()
number_halo_pileup_entries = 0

if halo_pileup_on:
    halo_pileup_tfile = ROOT.TFile.Open(halo_pileup_file, 'read')

    # check if TTree object exists
    if not halo_pileup_tfile.GetListOfKeys().Contains('halo_pileup'):
        print 'Cannot find TObject: halo_pileup'
        sys.exit(1)

    halo_pileup_ttree = halo_pileup_tfile.Get('halo_pileup')
    number_halo_pileup_entries = halo_pileup_ttree.GetEntries()

#/////////////////////////////////////////////////////////////
# for plotting
#/////////////////////////////////////////////////////////////

pdg_codes = [ int(pdg_code) for pdg_code in pdg_coefficients.keys() ]

beam_x_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_y_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
beam_z_dist = OrderedDict([ (pdg, []) for pdg in pdg_codes ])
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

#/////////////////////////////////////////////////////////////
# particle beam composition
#/////////////////////////////////////////////////////////////
beam_particle = hepevt.Particle(pdg_coefficients)

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
    beam_ttree.GetEntry(beam_idx)

    # initialize number of beam particles and halo pile-up particles
    number_beam = 1
    number_halo_pileup = 0

    # initialize strings for writing to output text file
    beam_str = ''
    halo_pileup_str = ''

    # if halo pile-up TTree object is present
    if halo_pileup_on and halo_pileup_ttree is not None:

        # draw random number for halo pile-up entry
        halo_pileup_idx = np.random.randint(0, number_halo_pileup_entries)
        halo_pileup_ttree.GetEntry(halo_pileup_idx)

        # get number of halo pile-up particles
        number_halo_pileup = halo_pileup_ttree.halo_pileup_number_particles

        for pileup_idx in xrange(number_halo_pileup):

            # get PDG code of halo_pileup particle
            halo_pileup_pdg_code_ = halo_pileup_pdg_codes[np.random.randint(0, len(halo_pileup_pdg_codes))]

            # get halo pile-up particle position
            halo_pileup_x_ = halo_pileup_ttree.halo_pileup_x[pileup_idx]
            halo_pileup_y_ = halo_pileup_ttree.halo_pileup_y[pileup_idx]
            halo_pileup_z_ = halo_pileup_ttree.halo_pileup_z[pileup_idx]

            # get halo-pileup particle angle
            halo_pileup_angle_xz_ = halo_pileup_ttree.halo_pileup_angle_xz[pileup_idx]
            halo_pileup_angle_yz_ = halo_pileup_ttree.halo_pileup_angle_yz[pileup_idx]

            # get halo_pileup particle momentum
            #halo_pileup_momentum_ = halo_pileup.halo_pileup_momentum[pileup_idx] / 1000.0
            halo_pileup_momentum_ = np.random.uniform(halo_pileup_momentum_low, halo_pileup_momentum_high)

            halo_pileup_momentum_x_, halo_pileup_momentum_y_, halo_pileup_momentum_z_ = hepevt.rotate(
                0, 0, halo_pileup_momentum_,
                halo_pileup_angle_xz_ * np.pi/180.0, halo_pileup_angle_yz_ * np.pi/180.0)

            # compute halo_pileup particle mass and energy
            halo_pileup_mass_ = particle_mass[np.abs(halo_pileup_pdg_code_)] / 1000.0
            halo_pileup_energy_ = np.sqrt(
                halo_pileup_momentum_ * halo_pileup_momentum_ + halo_pileup_mass_ * halo_pileup_mass_)

            # set halo_pileup particle time
            halo_pileup_time_ = 0.0

            #halo_pileup_pdg_code = np.array([
            #    halo_pileup_pdg_codes[idx] for idx in np.random.randint(
            #        0, len(halo_pileup_pdg_codes), number_halo_pileup)])

            # append to halo pile-up string
            halo_pileup_str += (
                '1' + ' ' +
                str(halo_pileup_pdg_code_)   + ' 0 0 0 0 ' +
                str(halo_pileup_momentum_x_) + ' ' +
                str(halo_pileup_momentum_y_) + ' ' +
                str(halo_pileup_momentum_z_) + ' ' +
                str(halo_pileup_energy_)     + ' ' +
                str(halo_pileup_mass_)       + ' ' +
                str(halo_pileup_x_)          + ' ' +
                str(halo_pileup_y_)          + ' ' +
                str(halo_pileup_z_)          + ' ' +
                str(halo_pileup_time_)       + '\n')

            halo_pileup_x_dist.append(halo_pileup_x_)
            halo_pileup_y_dist.append(halo_pileup_y_)
            halo_pileup_z_dist.append(halo_pileup_z_)
            halo_pileup_angle_xz_dist.append(halo_pileup_angle_xz_)
            halo_pileup_angle_yz_dist.append(halo_pileup_angle_yz_)
            halo_pileup_momentum_dist.append(halo_pileup_momentum_ * 1000.0)
            halo_pileup_momentum_x_dist.append(halo_pileup_momentum_x_ * 1000.0)
            halo_pileup_momentum_y_dist.append(halo_pileup_momentum_y_ * 1000.0)
            halo_pileup_momentum_z_dist.append(halo_pileup_momentum_z_ * 1000.0)

        halo_pileup_number_particles_dist.append(number_halo_pileup)

    # get total number of particles
    number_particles = number_beam + number_halo_pileup
    f.write(str(evt_idx) + ' ' + str(number_particles) + '\n')

    # get beam particle position
    beam_x_ = beam_ttree.beam_x
    beam_y_ = beam_ttree.beam_y
    beam_z_ = beam_ttree.beam_z

    # get beam particle angle
    beam_angle_xz_ = beam_ttree.beam_angle_xz
    beam_angle_yz_ = beam_ttree.beam_angle_yz

    # get beam particle momentum
    beam_momentum_ = beam_ttree.beam_momentum / 1000.0

    # get PDG code of beam particle
    #beam_pdg_code_ = beam_pdg_codes[np.random.randint(0, len(beam_pdg_codes))]
    beam_pdg_code_ = beam_particle.pdg_from_momentum(beam_momentum_ * 1000.0)

    beam_momentum_x_, beam_momentum_y_, beam_momentum_z_ = hepevt.rotate(
        0, 0, beam_momentum_,
        beam_angle_xz_ * np.pi/180.0, beam_angle_yz_ * np.pi/180.0)

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

    beam_x_dist[beam_pdg_code_].append(beam_x_)
    beam_y_dist[beam_pdg_code_].append(beam_y_)
    beam_z_dist[beam_pdg_code_].append(beam_z_)
    beam_angle_xz_dist[beam_pdg_code_].append(beam_angle_xz_)
    beam_angle_yz_dist[beam_pdg_code_].append(beam_angle_yz_)
    beam_momentum_dist[beam_pdg_code_].append(beam_momentum_ * 1000.0)
    beam_momentum_x_dist[beam_pdg_code_].append(beam_momentum_x_ * 1000.0)
    beam_momentum_y_dist[beam_pdg_code_].append(beam_momentum_y_ * 1000.0)
    beam_momentum_z_dist[beam_pdg_code_].append(beam_momentum_z_ * 1000.0)

    # if halo pile-up TTree object is present
    if halo_pileup_on and halo_pileup_ttree is not None and not not halo_pileup_str:
        f.write(halo_pileup_str)

beam_plotter = hepevt.Plotter('beam', config)

beam_plotter.plot_position(beam_x_dist, beam_y_dist, beam_z_dist)
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

