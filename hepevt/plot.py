import ConfigParser
import ast
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator

class Plotter:

    def __init__(self, prefix, config):

        self.prefix = prefix
        self.plot_output_dir = ast.literal_eval(config.get('config', 'plot_output_dir'))
        self.title_prefix = ast.literal_eval(config.get('config', '%s_title_prefix' % prefix))

        self.position_x_bins = config.getint('config', '%s_position_x_bins' % prefix)
        self.position_y_bins = config.getint('config', '%s_position_y_bins' % prefix)
        self.position_z_bins = config.getint('config', '%s_position_z_bins' % prefix)

        self.position_x_bin_low = config.getfloat('config', '%s_position_x_bin_low' % prefix)
        self.position_y_bin_low = config.getfloat('config', '%s_position_y_bin_low' % prefix)
        self.position_z_bin_low = config.getfloat('config', '%s_position_z_bin_low' % prefix)

        self.position_x_bin_high = config.getfloat('config', '%s_position_x_bin_high' % prefix)
        self.position_y_bin_high = config.getfloat('config', '%s_position_y_bin_high' % prefix)
        self.position_z_bin_high = config.getfloat('config', '%s_position_z_bin_high' % prefix)

        self.momentum_bins = config.getint('config', '%s_momentum_bins' % prefix)
        self.momentum_bin_low = config.getfloat('config', '%s_momentum_bin_low' % prefix)
        self.momentum_bin_high = config.getfloat('config', '%s_momentum_bin_high' % prefix)

        self.momentum_x_bins = config.getint('config', '%s_momentum_x_bins' % prefix)
        self.momentum_y_bins = config.getint('config', '%s_momentum_y_bins' % prefix)
        self.momentum_z_bins = config.getint('config', '%s_momentum_z_bins' % prefix)

        self.momentum_x_bin_low = config.getfloat('config', '%s_momentum_x_bin_low' % prefix)
        self.momentum_y_bin_low = config.getfloat('config', '%s_momentum_y_bin_low' % prefix)
        self.momentum_z_bin_low = config.getfloat('config', '%s_momentum_z_bin_low' % prefix)

        self.momentum_x_bin_high = config.getfloat('config', '%s_momentum_x_bin_high' % prefix)
        self.momentum_y_bin_high = config.getfloat('config', '%s_momentum_y_bin_high' % prefix)
        self.momentum_z_bin_high = config.getfloat('config', '%s_momentum_z_bin_high' % prefix)

        self.angle_xz_bins = config.getint('config', '%s_angle_xz_bins' % prefix)
        self.angle_yz_bins = config.getint('config', '%s_angle_yz_bins' % prefix)

        self.angle_xz_bin_low = config.getfloat('config', '%s_angle_xz_bin_low' % prefix)
        self.angle_yz_bin_low = config.getfloat('config', '%s_angle_yz_bin_low' % prefix)

        self.angle_xz_bin_high = config.getfloat('config', '%s_angle_xz_bin_high' % prefix)
        self.angle_yz_bin_high = config.getfloat('config', '%s_angle_yz_bin_high' % prefix)

        self.particle_bins = config.getint('config', '%s_particle_bins' % prefix)
        self.particle_bin_low = config.getfloat('config', '%s_particle_bin_low' % prefix)
        self.particle_bin_high = config.getfloat('config', '%s_particle_bin_high' % prefix)

        self.time_bins = config.getint('config', '%s_time_bins' % prefix)
        self.time_bin_low = config.getfloat('config', '%s_time_bin_low' % prefix)
        self.time_bin_high = config.getfloat('config', '%s_time_bin_high' % prefix)

    def plot_position(self, x, y, z):

        x_dist = []
        y_dist = []
        z_dist = []

        x_stack = False
        y_stack = False
        z_stack = False

        x_labels = []
        y_labels = []
        z_labels = []

        if isinstance(x, OrderedDict):
            x_stack = True
            for pdg, dist in x.items():
                x_dist.append(dist)
                x_labels.append(pdg)
            if len(x) > 1:
                x_dist = np.array(x_dist)
            else:
                x_dist = np.array(x_dist).flatten()
        else:
            x_dist = np.array(x)

        if isinstance(y, OrderedDict):
            y_stack = True
            for pdg, dist in y.items():
                y_dist.append(dist)
                y_labels.append(pdg)
            if len(y) > 1:
                y_dist = np.array(y_dist)
            else:
                y_dist = np.array(y_dist).flatten()
        else:
            y_dist = np.array(y)

        if isinstance(z, OrderedDict):
            z_stack = True
            for pdg, dist in z.items():
                z_dist.append(dist)
                z_labels.append(pdg)
            z_dist = np.array(z_dist)
            if len(z) > 1:
                z_dist = np.array(z_dist)
            else:
                z_dist = np.array(z_dist).flatten()
        else:
            z_dist = np.array(z)

        x_bins = self.position_x_bins
        x_range = (self.position_x_bin_low, self.position_x_bin_high)
        x_bin_width = (x_range[1] - x_range[0]) / x_bins

        y_bins = self.position_y_bins
        y_range = (self.position_y_bin_low, self.position_y_bin_high)
        y_bin_width = (y_range[1] - y_range[0]) / y_bins

        z_bins = self.position_z_bins
        z_range = (self.position_z_bin_low, self.position_z_bin_high)
        z_bin_width = (z_range[1] - z_range[0]) / z_bins

        title_prefix = self.title_prefix

        fig, axarr = plt.subplots(2, 2)

        axarr[0, 0].hist(x_dist, x_bins, range=(x_range[0], x_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=x_stack, label=x_labels)
        axarr[0, 0].set_title(r'%s $x$ distribution' % title_prefix)
        axarr[0, 0].set_xlabel(r'$x$ [cm]')
        axarr[0, 0].set_ylabel('entries / %s cm' % str(x_bin_width))
        axarr[0, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[0, 0].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[0, 0].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)
        if x_stack:
            axarr[0, 0].legend(prop={'size': 8}, loc='upper left')

        axarr[0, 1].hist(y_dist, y_bins, range=(y_range[0], y_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=y_stack, label=y_labels)
        axarr[0, 1].set_title(r'%s $y$ distribution' % title_prefix)
        axarr[0, 1].set_xlabel(r'$y$ [cm]')
        axarr[0, 1].set_ylabel('entries / %s cm' % str(y_bin_width))
        axarr[0, 1].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[0, 1].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[0, 1].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)

        axarr[1, 0].hist(z_dist, z_bins, range=(z_range[0], z_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=z_stack, label=z_labels)
        axarr[1, 0].set_title(r'%s $z$ distribution' % title_prefix)
        axarr[1, 0].set_xlabel(r'$z$ [cm]')
        axarr[1, 0].set_ylabel('entries / %s cm' % z_bin_width)
        axarr[1, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[1, 0].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[1, 0].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)
        #if z_stack:
        #    axarr[1, 0].legend(prop={'size': 8})
        axarr[1, 0].set_xlim(-20, 160)
        #axarr[1, 0].set_ylim(0, 5000)

        hist, xbins, ybins = np.histogram2d(np.hstack(x_dist), np.hstack(y_dist),
                                            bins=[ x_bins, y_bins ],
                                            range=(x_range, y_range))
        extent = [ xbins.min(), xbins.max(), ybins.min(), ybins.max() ]
        im = axarr[1, 1].imshow(np.ma.masked_where(hist == 0, hist).T,
                                interpolation='nearest', origin='lower',
                                extent=extent)
        axarr[1, 1].set_title(r'%s $xy$ distribution' % title_prefix)
        axarr[1, 1].set_xlabel(r'$x$ [cm]')
        axarr[1, 1].set_ylabel(r'$y$ [cm]')
        axarr[1, 1].set_xlim(x_range)
        axarr[1, 1].set_ylim(y_range)
        axarr[1, 1].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[1, 1].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[1, 1].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)
        color_bar = fig.colorbar(im)
        color_bar.set_label(r'entries per %s cm $\times$ %s cm' % (x_bin_width, y_bin_width))
        #color_bar.set_label(r'entries / cm$^2$')

        plt.tight_layout()

        #plt.show()
        plt.savefig(self.plot_output_dir + self.prefix + '_position.pdf')

    def plot_momentum(self, p, px, py, pz, angle_xz, angle_yz):

        p_dist = []
        px_dist = []
        py_dist = []
        pz_dist = []

        p_stack = False
        px_stack = False
        py_stack = False
        pz_stack = False

        p_labels = []
        px_labels = []
        py_labels = []
        pz_labels = []

        angle_xz_dist = []
        angle_yz_dist = []

        angle_xz_stack = False
        angle_yz_stack = False

        angle_xz_labels = []
        angle_yz_labels = []

        if isinstance(p, OrderedDict):
            p_stack = True
            for pdg, dist in p.items():
                p_dist.append(dist)
                p_labels.append(pdg)
            if len(p) > 1:
                p_dist = np.array(p_dist)
            else:
                p_dist = np.array(p_dist).flatten()
        else:
            p_dist = np.array(p)

        if isinstance(px, OrderedDict):
            px_stack = True
            for pdg, dist in px.items():
                px_dist.append(dist)
                px_labels.append(pdg)
            if len(px) > 1:
                px_dist = np.array(px_dist)
            else:
                px_dist = np.array(px_dist).flatten()
        else:
            px_dist = np.array(px)

        if isinstance(py, OrderedDict):
            py_stack = True
            for pdg, dist in py.items():
                py_dist.append(dist)
                py_labels.append(pdg)
            if len(py) > 1:
                py_dist = np.array(py_dist)
            else:
                py_dist = np.array(py_dist).flatten()
        else:
            py_dist = np.array(py)

        if isinstance(pz, OrderedDict):
            pz_stack = True
            for pdg, dist in pz.items():
                pz_dist.append(dist)
                pz_labels.append(pdg)
            if len(pz) > 1:
                pz_dist = np.array(pz_dist)
            else:
                pz_dist = np.array(pz_dist).flatten()
        else:
            pz_dist = np.array(pz)

        if isinstance(angle_xz, OrderedDict):
            angle_xz_stack = True
            for pdg, dist in angle_xz.items():
                angle_xz_dist.append(dist)
                angle_xz_labels.append(pdg)
            if len(angle_xz) > 1:
                angle_xz_dist = np.array(angle_xz_dist)
            else:
                angle_xz_dist = np.array(angle_xz_dist).flatten()
        else:
            angle_xz_dist = np.array(angle_xz)

        if isinstance(angle_yz, OrderedDict):
            angle_yz_stack = True
            for pdg, dist in angle_yz.items():
                angle_yz_dist.append(dist)
                angle_yz_labels.append(pdg)
            if len(angle_yz) > 1:
                angle_yz_dist = np.array(angle_yz_dist)
            else:
                angle_yz_dist = np.array(angle_yz_dist).flatten()
        else:
            angle_yz_dist = np.array(angle_yz)

        p_bins = self.momentum_bins
        p_range = (self.momentum_bin_low, self.momentum_bin_high)
        p_bin_width = (p_range[1] - p_range[0]) / p_bins

        px_bins = self.momentum_x_bins
        px_range = (self.momentum_x_bin_low, self.momentum_x_bin_high)
        px_bin_width = (px_range[1] - px_range[0]) / px_bins

        py_bins = self.momentum_y_bins
        py_range = (self.momentum_y_bin_low, self.momentum_y_bin_high)
        py_bin_width = (py_range[1] - py_range[0]) / py_bins

        pz_bins = self.momentum_z_bins
        pz_range = (self.momentum_z_bin_low, self.momentum_z_bin_high)
        pz_bin_width = (pz_range[1] - pz_range[0]) / pz_bins

        angle_xz_bins = self.angle_xz_bins
        angle_xz_range = (self.angle_xz_bin_low, self.angle_xz_bin_high)
        angle_xz_bin_width = (angle_xz_range[1] - angle_xz_range[0]) / angle_xz_bins

        angle_yz_bins = self.angle_yz_bins
        angle_yz_range = (self.angle_yz_bin_low, self.angle_yz_bin_high)
        angle_yz_bin_width = (angle_yz_range[1] - angle_yz_range[0]) / angle_yz_bins

        title_prefix = self.title_prefix

        fig, axarr = plt.subplots(2, 3, figsize=(8, 5))

        axarr[0, 0].hist(px_dist, px_bins, range=(px_range[0], px_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=px_stack, label=px_labels)
        axarr[0, 0].set_title(r'%s $p_x$ distribution' % title_prefix)
        axarr[0, 0].set_xlabel(r'$p_x$ [MeV/c]')
        axarr[0, 0].set_ylabel('entries / %s MeV/c' % str(px_bin_width))
        axarr[0, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[0, 0].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[0, 0].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)
        if px_stack:
            axarr[0, 0].legend(prop={'size': 8})

        axarr[0, 1].hist(py_dist, py_bins, range=(py_range[0], py_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=py_stack, label=py_labels)
        axarr[0, 1].set_title(r'%s $p_y$ distribution' % title_prefix)
        axarr[0, 1].set_xlabel(r'$p_y$ [MeV/c]')
        axarr[0, 1].set_ylabel('entries / %s MeV/c' % str(py_bin_width))
        axarr[0, 1].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[0, 1].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[0, 1].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)

        axarr[0, 2].hist(pz_dist, pz_bins, range=(pz_range[0], pz_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=pz_stack, label=pz_labels)
        axarr[0, 2].set_title(r'%s $p_z$ distribution' % title_prefix)
        axarr[0, 2].set_xlabel(r'$p_z$ [MeV/c]')
        axarr[0, 2].set_ylabel('entries / %s MeV/c' % pz_bin_width)
        axarr[0, 2].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[0, 2].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[0, 2].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)

        axarr[1, 0].hist(angle_xz_dist, angle_xz_bins,
                         range=(angle_xz_range[0], angle_xz_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=angle_xz_stack, label=angle_xz_labels)
        axarr[1, 0].set_title(r'%s $\theta_{xz}$ distribution' % title_prefix)
        axarr[1, 0].set_xlabel(r'$\theta_{xz}$ [$^\circ$]')
        axarr[1, 0].set_ylabel('entries / %s$^\circ$' % angle_xz_bin_width)
        axarr[1, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[1, 0].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[1, 0].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)
        axarr[1, 0].set_yscale('log', nonposy='clip')

        axarr[1, 1].hist(angle_yz_dist, angle_yz_bins,
                         range=(angle_yz_range[0], angle_yz_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=angle_yz_stack, label=angle_yz_labels)
        axarr[1, 1].set_title(r'%s $\theta_{yz}$ distribution' % title_prefix)
        axarr[1, 1].set_xlabel(r'$\theta_{yz}$ [$^\circ$]')
        axarr[1, 1].set_ylabel('entries / %s$^\circ$' % angle_yz_bin_width)
        axarr[1, 1].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[1, 1].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[1, 1].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)
        axarr[1, 1].set_yscale('log', nonposy='clip')

        axarr[1, 2].hist(p_dist, p_bins, range=(p_range[0], p_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=p_stack, label=p_labels)
        axarr[1, 2].set_title(r'%s $p$ distribution' % title_prefix)
        axarr[1, 2].set_xlabel(r'$p$ [MeV/c]')
        axarr[1, 2].set_ylabel('entries / %s MeV/c' % str(p_bin_width))
        axarr[1, 2].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[1, 2].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[1, 2].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)

        plt.tight_layout()

        #plt.show()
        plt.savefig(self.plot_output_dir + self.prefix + '_momentum.pdf')

    def plot_number_particles(self, x):

        #x_dist = []

        #x_stack = False

        #x_labels = []

        #if isinstance(x, OrderedDict):
        #    x_stack = True
        #    for pdg, dist in x.items():
        #        x_dist.append(dist)
        #        x_labels.append(pdg)
        #    x_dist = np.array(x_dist)
        #else:
        #    x_dist = np.array(x)

        particle_bins = self.particle_bins
        particle_range = (self.particle_bin_low, self.particle_bin_high)
        particle_bin_width = (particle_range[1] - particle_range[0]) / particle_bins

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        n, bins, patches = ax.hist(x, particle_bins,
                                   range=(particle_range[0], particle_range[1]),
                                   alpha=0.75, histtype='stepfilled')

        ax.set_xlabel('number of incident %s particles' % self.title_prefix)
        ax.set_ylabel('entries')

        #ax.xaxis.set_minor_locator(AutoMinorLocator())
        #ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        #ax.grid(True, which='both', axis='both', color='k', linestyle=':',
        #        linewidth=1, alpha=0.2)

        ax.set_xlim([particle_range[0], particle_range[1]])

        #plt.show()
        plt.savefig(self.plot_output_dir + self.prefix + '_number_particles.pdf')

    def plot_time_offset(self, x0, y0, t0, x1):

        x0_dist = []
        y0_dist = []
        t0_dist = []
        x1_dist = []

        x0_stack = False
        y0_stack = False
        t0_stack = False
        x1_stack = False

        x0_labels = []
        y0_labels = []
        t0_labels = []
        x1_labels = []

        if isinstance(x0, OrderedDict):
            x0_stack = True
            for pdg, dist in x0.items():
                x0_dist.append(dist)
                x0_labels.append(pdg)
            if len(x0) > 1:
                x0_dist = np.array(x0_dist)
            else:
                x0_dist = np.array(x0_dist).flatten()
        else:
            x0_dist = np.array(x0)

        if isinstance(y0, OrderedDict):
            y0_stack = True
            for pdg, dist in y0.items():
                y0_dist.append(dist)
                y0_labels.append(pdg)
            if len(y0) > 1:
                y0_dist = np.array(y0_dist)
            else:
                y0_dist = np.array(y0_dist).flatten()
        else:
            y0_dist = np.array(y0)

        if isinstance(t0, OrderedDict):
            t0_stack = True
            for pdg, dist in t0.items():
                t0_dist.append(dist)
                t0_labels.append(pdg)
            if len(t0) > 1:
                t0_dist = np.array(t0_dist)
            else:
                t0_dist = np.array(t0_dist).flatten()
        else:
            t0_dist = np.array(t0)

        if isinstance(x1, OrderedDict):
            x1_stack = True
            for pdg, dist in x1.items():
                x1_dist.append(dist)
                x1_labels.append(pdg)
            if len(x1) > 1:
                x1_dist = np.array(x1_dist)
            else:
                x1_dist = np.array(x1_dist).flatten()
        else:
            x1_dist = np.array(x1)

        x0_bins = self.position_x_bins
        x0_range = (self.position_x_bin_low, self.position_x_bin_high)
        x0_bin_width = (x0_range[1] - x0_range[0]) / x0_bins

        y0_bins = self.position_y_bins
        y0_range = (self.position_y_bin_low, self.position_y_bin_high)
        y0_bin_width = (y0_range[1] - y0_range[0]) / y0_bins

        t0_bins = self.time_bins
        t0_range = (self.time_bin_low, self.time_bin_high)
        t0_bin_width = (t0_range[1] - t0_range[0]) / t0_bins

        x1_bins = self.position_x_bins
        x1_range = (self.position_x_bin_low, self.position_x_bin_high)
        x1_bin_width = (x1_range[1] - x1_range[0]) / x1_bins

        title_prefix = self.title_prefix

        fig, axarr = plt.subplots(2, 2)

        axarr[0, 0].hist(x0_dist, x0_bins, range=(x0_range[0], x0_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=x0_stack, label=x0_labels)
        axarr[0, 0].set_title(r'%s $x_0$ distribution' % title_prefix)
        axarr[0, 0].set_xlabel(r'$x_0$ [cm]')
        axarr[0, 0].set_ylabel('entries / %s cm' % str(x0_bin_width))
        axarr[0, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[0, 0].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[0, 0].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)

        axarr[0, 1].hist(t0_dist, t0_bins, range=(t0_range[0], t0_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=t0_stack, label=t0_labels)
        axarr[0, 1].set_title(r'%s $t_0$ distribution' % title_prefix)
        axarr[0, 1].set_xlabel(r'$t_0$ [$\mu$s]')
        axarr[0, 1].set_ylabel('entries / %s $\mu$s' % str(t0_bin_width))
        axarr[0, 1].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[0, 1].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[0, 1].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)

        axarr[1, 0].hist(x1_dist, x1_bins, range=(x1_range[0], x1_range[1]),
                         alpha=0.75, histtype='stepfilled',
                         stacked=x1_stack, label=x1_labels)
        axarr[1, 0].set_title(r'%s $x_1$ distribution' % title_prefix)
        axarr[1, 0].set_xlabel(r'$x_1$ [cm]')
        axarr[1, 0].set_ylabel('entries / %s cm' % str(x1_bin_width))
        axarr[1, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[1, 0].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[1, 0].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)

        hist, xbins, ybins = np.histogram2d(np.hstack(x0_dist), np.hstack(y0_dist),
                                            bins=[ x0_bins, y0_bins ],
                                            range=(x0_range, y0_range))
        extent = [ xbins.min(), xbins.max(), ybins.min(), ybins.max() ]
        im = axarr[1, 1].imshow(np.ma.masked_where(hist == 0, hist).T,
                                interpolation='nearest', origin='lower',
                                extent=extent)
        axarr[1, 1].set_title(r'%s $x_0 y_0$ distribution' % title_prefix)
        axarr[1, 1].set_xlabel(r'$x_0$ [cm]')
        axarr[1, 1].set_ylabel(r'$y_0$ [cm]')
        axarr[1, 1].set_xlim(x0_range)
        axarr[1, 1].set_ylim(y0_range)
        axarr[1, 1].xaxis.set_minor_locator(AutoMinorLocator())
        axarr[1, 1].yaxis.set_minor_locator(AutoMinorLocator())
        #axarr[1, 1].grid(True, which='both', axis='both', color='k',
        #                 linestyle=':', linewidth=1, alpha=0.2)
        color_bar = fig.colorbar(im)
        color_bar.set_label(r'entries per %s cm $\times$ %s cm' % (x0_bin_width, y0_bin_width))

        plt.tight_layout()
        plt.savefig(self.plot_output_dir + self.prefix + '_time_offset.pdf')

