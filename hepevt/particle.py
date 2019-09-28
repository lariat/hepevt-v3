import numpy as np

class Particle:
    """ Particle class. """

    def __init__(self, pdg_coefficients):
        self.pdg_coefficients = pdg_coefficients
        self.polynomials = {
            int(pdg_code) : np.poly1d(pdg_coefficients[pdg_code])
            for pdg_code in pdg_coefficients.keys()
            }

    def pdg_from_momentum(self, momentum):

        pdg = []
        probability = []

        for pdg_code, polynomial in self.polynomials.iteritems():
            pdg.append(pdg_code)
            probability.append(polynomial(momentum))
            #print polynomial(momentum)
            #if np.abs(pdg_code) == 11:
            #    probability[-1] += 0.2
            #if np.abs(pdg_code) == 13:
            #    probability[-1] += 0.3

        probability = np.array(probability)
        probability[probability < 0] = 0
        probability /= sum(probability)

        #print probability

        return np.random.choice(pdg, p=probability)

        particle_pdg = np.random.choice(pdg, p=probability)

        return particle_pdg

if __name__ == '__main__':

    # -100 A
    #coefficients = {
    #    -211 : [ -9.00471e-07,  0.00156373 , 0.135303 ],
    #      13 : [  1.59299e-07, -0.000270598, 0.234394 ],
    #      11 : [  2.84641e-07, -0.000621823, 0.385299 ],
    #    }

    # new -100A coefficients from Greg; 2019-02-12
    #coefficients = {
    #    -211 : [ -8.84e-08,  0.000353 , 0.671 ],
    #      13 : [ -1.29e-07,  0.000163, -0.013 ],
    #      11 : [  1.37e-07, -0.000408,  0.302 ],
    #    }

    # -100A coefficients, shit fit; 2019-09-27
    # coefficients = {
    #     -211 : [ -4.475927306880658e-07 ,  0.0009023390738552262,  0.43708484235478634 ],
    #       13 : [  2.8566942286599146e-07, -0.0004729858285416102,  0.2689511173523179  ],
    #       11 : [  3.2439077928434735e-07, -0.0006789612927594245,  0.387388718405574   ],
    #     }

    ## +60 A
    #coefficients = {
    #     211 : [ -2.34065e-06,  0.00303159 , -0.297014 ],
    #     -13 : [  5.90684e-07, -0.000394666,  0.189165 ],
    #     -11 : [  1.61466e-06, -0.00253278 ,  1.08608  ],
    #    }

    # -100A coefficients; 2019-09-27
    coefficients = {
        -211 : [ -6.91456e-07,  0.00127661 ,  0.296662 ],
          13 : [  1.65673e-07, -0.000284906,  0.195715 ],
          11 : [  2.55628e-07, -0.000580248,  0.351821 ],
        }

    # # -60A coefficients; 2019-09-27
    # coefficients = {
    #     -211 : [ -2.83885e-06,  0.00380683 , -0.446725 ],
    #       13 : [  4.22408e-07, -0.000355842,  0.153384 ],
    #       11 : [  2.34913e-06, -0.00335186 ,  1.25818  ],
    #     }

    # # +100A coefficients; 2019-09-27
    # coefficients = {
    #      211 : [ -6.40252e-07,  0.00119789 ,  0.337248 ],
    #      -13 : [  2.39278e-07, -0.000399755,  0.240828 ],
    #      -11 : [  2.39754e-07, -0.000535427,  0.315675 ],
    #     }

    # # +60A coefficients; 2019-09-27
    # coefficients = {
    #      211 : [ -2.37836e-06,  0.00318027 , -0.2419   ],
    #      -13 : [  2.05026e-07, -0.000149928,  0.106633 ],
    #      -11 : [  1.747e-06  , -0.00259285 ,  1.02186  ],
    #     }

    particle = Particle(coefficients)

    #for momentum in np.linspace(200, 1000-50, 16):
    for momentum in np.linspace(0, 1500-50, 30):

        particle_pdg_list = []

        for evt_idx in xrange(10000):
            particle_pdg = particle.pdg_from_momentum(momentum)
            particle_pdg_list.append(particle_pdg)

        unique, counts = np.unique(particle_pdg_list, return_counts=True)

        counts_ = np.array(counts, dtype=np.float) / np.sum(counts)

        print str(momentum) + ' MeV/c', dict(zip(unique, counts_)), np.sum(counts_)

