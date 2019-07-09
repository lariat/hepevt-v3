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
    coefficients = {
        -211 : [ -8.84e-08,  0.000353 , 0.671 ],
          13 : [ -1.29e-07,  0.000163, -0.013 ],
          11 : [  1.37e-07, -0.000408,  0.302 ],
        }

    ## +60 A
    #coefficients = {
    #    -211 : [ -2.34065e-06,  0.00303159 , -0.297014 ],
    #      13 : [  5.90684e-07, -0.000394666,  0.189165 ],
    #      11 : [  1.61466e-06, -0.00253278 ,  1.08608  ],
    #    }

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

