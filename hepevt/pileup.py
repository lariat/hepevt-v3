import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import scipy.interpolate as interpolate
#import scipy.stats
#from scipy.stats import poisson as poisson

# inverse transform sample
def inv_transform_sample(bin_counts, bin_edges, number_samples):
    bin_counts /=  np.sum(bin_counts)
    hist = bin_counts / np.diff(bin_edges)[0]
    cumulative_values = np.zeros(bin_edges.shape)
    cumulative_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cumulative_values, bin_edges)
    #plt.plot(bin_edges, cumulative_values)
    if not isinstance(number_samples, int):
        number_samples = int(number_samples)
    r = np.random.rand(number_samples)
    return inv_cdf(r)

# get number of halo pile-up particles
def get_number_halo_pileup_particles(bin_counts_txt_file, bin_edges_txt_file,
                                     parameters, number_events,
                                     tracking_efficiency=1.0):

    bin_counts = np.loadtxt(bin_counts_txt_file)
    bin_edges = np.loadtxt(bin_edges_txt_file)

    number_bins = bin_edges.size - 1
    bin_low = np.min(bin_edges)
    bin_high = np.max(bin_edges)
    bin_width = (bin_high - bin_low) / number_bins

    #fig = plt.figure(figsize=(8, 5))
    #ax1 = fig.add_subplot(1, 1, 1)

    #ax1.step(bin_edges[1:], bin_counts, where='pre')
    ##ax1.step(bin_edges[:-1], bin_counts, where='post')
    #ax1.fill_between(bin_edges[1:], bin_counts, step="pre", alpha=0.4)

    #print(np.sum(bin_counts))

    sample = inv_transform_sample(bin_counts, bin_edges, number_events)
    #ax1.hist(sample, bins=number_bins, range=(bin_low, bin_high))

    p = np.poly1d(parameters)

    #x = bin_edges
    #y = p(bin_edges)

    #ax2 = ax1.twinx()

    #ax2.plot(x, y, 'k')

    #ax1.set_ylim(bottom=0)
    #ax2.set_ylim(bottom=0)

    #plt.show()

    mu = p(sample)
    mu /= tracking_efficiency

    #print mu
    #print np.random.poisson(mu)

    return np.random.poisson(mu)

class PileUp:
    """ PileUp class. """

    def __init__(self, bin_counts_txt_file_list, bin_edges_txt_file_list,
                 parameters_list, number_events, tracking_efficiency=1.0):

        if not (len(parameters_list) == len(bin_counts_txt_file_list) == len(bin_edges_txt_file_list)):
            raise ValueError("Iterables have different lengths")

        self.bin_counts_txt_file_list = bin_counts_txt_file_list
        self.bin_edges_txt_file_list = bin_edges_txt_file_list
        self.parameters_list = parameters_list
        self.number_events = number_events
        self.tracking_efficiency = tracking_efficiency

        bin_counts_integral = []

        for bin_counts_txt_file in self.bin_counts_txt_file_list:
            bin_counts = np.loadtxt(bin_counts_txt_file)
            bin_counts_sum = np.sum(bin_counts)
            bin_counts_integral.append(bin_counts_sum)

        p = np.array(bin_counts_integral).astype(float) / np.sum(bin_counts_integral)

        indices = np.random.choice(np.arange(len(p)), number_events, p=p)
        unique, counts = np.unique(indices, return_counts=True)
        iterable = dict(zip(unique, counts))

        number_halo_pileup_particles = []

        for idx, events in iterable.iteritems():

            number_halo_pileup_particles.append(
                get_number_halo_pileup_particles(
                    self.bin_counts_txt_file_list[idx],
                    self.bin_edges_txt_file_list[idx],
                    self.parameters_list[idx],
                    events,
                    tracking_efficiency)
                )

        self.number_halo_pileup_particles = np.hstack(number_halo_pileup_particles)

    def pdg_from_momentum(self, momentum):

        pdg = []
        probability = []

        for pdg_code, polynomial in self.polynomials.iteritems():
            pdg.append(pdg_code)
            probability.append(polynomial(momentum))
            #print polynomial(momentum)

        probability = np.array(probability)
        probability[probability < 0] = 0
        probability /= sum(probability)

        #print probability

        return np.random.choice(pdg, p=probability)

        particle_pdg = np.random.choice(pdg, p=probability)

        return particle_pdg

if __name__ == '__main__':

    tracking_efficiency = 0.56
    #number_events = 1000000
    number_events = 82814 + 67200

    # -100A

    #parameters_list = [
    #    [ 9.1915e-10, 2.3607e-01 ],
    #    ]

    #bin_counts_txt_file_list = [
    #    '../data/negative_100a_low_intensity_bin_counts.txt',
    #    ]

    #bin_edges_txt_file_list = [
    #    '../data/negative_100a_low_intensity_bin_edges.txt',
    #    ]

    # +100A

    parameters_list = [
        [ 2.13992368e-10, 1.75054177e-01 ],
        [ 7.84259587e-10, 2.14548821e-01 ],
        ]

    bin_counts_txt_file_list = [
        '../data/positive_100a_high_intensity_bin_counts.txt',
        '../data/positive_100a_low_intensity_bin_counts.txt',
        ]

    bin_edges_txt_file_list = [
        '../data/positive_100a_high_intensity_bin_edges.txt',
        '../data/positive_100a_low_intensity_bin_edges.txt',
        ]

    pileup = PileUp(bin_counts_txt_file_list, bin_edges_txt_file_list,
                    parameters_list, number_events, tracking_efficiency)

    print pileup.number_halo_pileup_particles

