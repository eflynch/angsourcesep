import numpy as np
import dspy
import matplotlib.pyplot as plt

# Additions to DSPY for Mixing Tracks
class Mixer(dspy.BundleGenerator):
    def __init__(self, generators, mixes):
        self._mixes = mixes
        dspy.BundleGenerator.__init__(self, generators)
        self.num_channels = 2

    def _generate(self, frame_count):
        output = np.zeros(frame_count * self.num_channels, dtype=np.float32)

        for m, g in zip(self._mixes, self.generators):
            data, cf = g.generate(frame_count)
            # Enfore mono sources
            buf = dspy.lib.rechannel(data, g.num_channels, 1)
            # Mix sources
            output[0::2] += m * buf
            output[1::2] += (1-m) * buf

        return output

# Mix signals with pan params a and b
def ABtoLR(sig, a, b):
    A = sig[0::2]
    B = sig[1::2]

    L = a*A + b*B
    R = (1-a) * A + (1-b)*B

    output = np.zeros(len(sig))
    output[0::2] = L
    output[1::2] = R

    return output

# Separate signals with pan params a and b
def LRtoAB(sig, a, b):
    if a == b:
        raise Exception("Matrix is singular for a==b!!")
    L = sig[0::2]
    R = sig[1::2]


    A = (1 - b)/(a-b) * L - b/(a-b) * R
    B = (a - 1)/(a-b) * L + a/(a-b) * R

    output = np.zeros(len(sig))
    output[0::2] = A
    output[1::2] = B

    return output

# Spectral Separation Metric
def spec_separation(sig, num_bins=100, show_plt=False):
    A = sig[0::2]
    B = sig[1::2]

    fftA = np.log(np.abs(np.fft.fft(A))[len(A)/2:])
    fftB = np.log(np.abs(np.fft.fft(B))[len(B)/2:])

    bins = np.linspace(0, len(fftA), num_bins)
    binsA = np.zeros(num_bins)
    binsB = np.zeros(num_bins)

    for i in xrange(len(bins) - 1):
        binsA[i] = np.sum(fftA[bins[i]:bins[i+1]])
        binsB[i] = np.sum(fftB[bins[i]:bins[i+1]])

    binsA = binsA / np.sqrt(binsA.dot(binsA))
    binsB = binsB / np.sqrt(binsB.dot(binsB))
    if show_plt:
        fig, axes = plt.subplots(1, 2)
        ax0, ax1 = axes.flat
        ax0.plot(fftA)
        ax0.plot(fftB)
        ax1.plot(binsA)
        ax1.plot(binsB)
        plt.show()

    return (binsA - binsB).dot(binsA - binsB)


def separate(lr, metric, n=20):
    sep = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            if (i == j):
                continue
            b = i / float(n)
            a = j / float(n)
            ab = LRtoAB(lr, a, b)
            sep[i,j] = metric(ab)


    amax = np.unravel_index(sep.argmax(), sep.shape)

    return (amax[0]/float(n), amax[1]/float(n)), sep

def separate2(lr, metric, n=20):
    sepA = np.zeros(n)
    for i in xrange(1,n):
        a = 0
        b = i / float(n)
        ab = LRtoAB(lr, a, b)
        sepA[i] = metric(ab)

    sepB = np.zeros(n)
    for i in xrange(0,n-1):
        a = 1
        b = i / float(n)
        ab = LRtoAB(lr, a, b)
        sepB[i] = metric(ab)

    peaksA = np.r_[False, sepA[1:] > sepA[:-1]] & np.r_[sepA[:-1] > sepA[1:], False]
    peaksB = np.r_[False, sepB[1:] > sepB[:-1]] & np.r_[sepB[:-1] > sepB[1:], False]
    domain = np.linspace(0,1,len(sepA),False)
    peaks = [domain[i] for i,f in enumerate(peaksA | peaksB) if f]

    if len(peaks) == 0:
        ab = (0,0)
    elif len(peaks) == 1:
        ab = (peaks[0],peaks[0])
    else:
        ab = peaks[:2]

    return ab, sepA, sepB


def plot_separation(sep_graph, title):
    sep_graph = np.flipud(sep_graph)
    fig, ax = plt.subplots()
    ax.set_title('Separation Graph for %s' % title)
    cax = ax.imshow(sep_graph, extent=[0, 1, 0, 1], interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    plt.savefig('experiment1/%s.png' % title)
    plt.close()


def run_experiment1(lr, alpha, beta):
    n = 20
    pan, graph = separate(lr, lambda sig: spec_separation(sig, 300, False), n)
    print "Expected (%.2f, %.2f); Got: (%.2f, %.2f)" % (alpha, beta, pan[0], pan[1])
    plot_separation(graph, '%.2f_%.2f' % (alpha, beta))
    return pan

def run_experiment2(lr, alpha, beta):
    n = 20
    pan, sepA, sepB = separate2(lr, lambda sig: spec_separation(sig, 300, False), n)
    print "Expected (%.2f, %.2f); Got: (%.2f, %.2f)" % (alpha, beta, pan[0], pan[1])
    plt.figure()
    plt.plot(np.linspace(0,1,len(sepA),False), sepA)
    plt.plot(np.linspace(0,1,len(sepB),False), sepB)
    plt.savefig('experiment2/%.2f_%.2f.png' % (alpha, beta))
    plt.close()
    return pan


if __name__=="__main__":
    # Mix Sources
    violinSampler = dspy.Sampler("./violin.wav")
    polishSampler = dspy.Sampler("./polish.wav")
    A = violinSampler.sample(0, 44100*3, True, 1.0)
    B = polishSampler.sample(0, 44100*3, True, 1.0)

    success = []
    failures = []
    for i in xrange(9):
        for j in xrange(9):
            alpha = 0.1 * (i+1)
            beta = 0.1 * (j+1)
            stereoMix = Mixer([A, B], [alpha, beta])
            lr, fc = stereoMix.generate(44100*5)
            # pan = run_experiment1(lr, alpha, beta)
            pan = run_experiment2(lr, alpha, beta)
            if (sorted(pan)[0]-sorted([alpha,beta])[0]) < 0.01 and (sorted(pan)[1]-sorted([alpha,beta])[1]) < 0.01:
                success.append((alpha, beta))
            else:
                failures.append((alpha, beta))

    print "Succeeded in ", len(success), "out of", len(success)+len(failures)

    

