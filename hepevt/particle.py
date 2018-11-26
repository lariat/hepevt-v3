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

        probability = np.array(probability)
        probability /= sum(probability)

        return np.random.choice(pdg, p=probability)

        particle_pdg = np.random.choice(pdg, p=probability)

        return particle_pdg

if __name__ == '__main__':

    # -100 A
    coefficients = {
        -211 : [ -9.00471e-07,  0.00156373 , 0.135303 ],
          13 : [  1.59299e-07, -0.000270598, 0.234394 ],
          11 : [  2.84641e-07, -0.000621823, 0.385299 ],
        }

    ## +60 A
    #coefficients = {
    #    -211 : [ -2.34065e-06,  0.00303159 , -0.297014 ],
    #      13 : [  5.90684e-07, -0.000394666,  0.189165 ],
    #      11 : [  1.61466e-06, -0.00253278 ,  1.08608  ],
    #    }

    particle = Particle(coefficients)

    for momentum in np.linspace(200, 1000-50, 16):

        particle_pdg_list = []

        for evt_idx in xrange(10000):
            particle_pdg = particle.pdg_from_momentum(momentum)
            particle_pdg_list.append(particle_pdg)

        unique, counts = np.unique(particle_pdg_list, return_counts=True)

        counts_ = np.array(counts, dtype=np.float) / np.sum(counts)

        print str(momentum) + ' MeV/c', dict(zip(unique, counts_))

