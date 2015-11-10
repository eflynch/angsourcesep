import numpy as np
import dspy
import matplotlib.pyplot as plt

class Mixer(dspy.BundleGenerator):
    def __init__(self, generators, mixes):
        self._mixes = mixes
        dspy.BundleGenerator.__init__(self, generators)
        self.num_channels = 2

    def _generate(self, frame_count):
        output = np.zeros(frame_count * self.num_channels, dtype=np.float32)

        for m, g in zip(self._mixes, self.generators):
            data, cf = g.generate(frame_count)
            output[0::2] += m * data
            output[1::2] += (1-m) * data

        return output

class Monofy(dspy.WrapperGenerator):
    def __init__(self, generator):
        dspy.WrapperGenerator.__init__(self, generator)
        self.num_channels = 1

    def _generate(self, frame_count):
        buf, cf = self.generator.generate(frame_count)
        return dspy.lib.rechannel(buf, self.generator.num_channels, 1)


def LRtoAB(sig, a, b):
    if a == b:
        raise Exception("Matrix is singular for a==b")
    L = sig[0::2]
    R = sig[1::2]


    A = (1 - b)/(a-b) * L - b/(a-b) * R
    B = (a - 1)/(a-b) * L + a/(a-b) * R

    output = np.zeros(len(sig))
    output[0::2] = A
    output[1::2] = B

    return output

def ABtoLR(sig, a, b):
    A = sig[0::2]
    B = sig[1::2]

    L = a*A + b*B
    R = (1-a) * A + (1-b)*B

    output = np.zeros(len(sig))
    output[0::2] = L
    output[1::2] = R
    return output

def separation(sig, num_bins=100, show_plt=False):
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

    # return np.arccos(np.dot(binsA, binsB))

def separate(lr, n=20):
    sep = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            if (i == j):
                continue
            a = i / float(n)
            b = j / float(n)
            ab = LRtoAB(lr, a, b)
            sep[i,j] = separation(ab, 300)

    plt.figure()
    plt.imshow(sep, interpolation='nearest')
    plt.colorbar()
    plt.show()


if __name__=="__main__":
    alpha = 0.8
    beta = 0.2

    # Mix Sources
    violinSampler = dspy.Sampler("./violin.wav")
    polishSampler = dspy.Sampler("./polish.wav")
    A = Monofy(violinSampler.sample(0, 44100*3, True, 1.0))
    B = Monofy(polishSampler.sample(0, 44100*3, True, 1.0))
    stereoMix = Mixer([A, B], [alpha, beta])

    lr, fc = stereoMix.generate(44100*5)

    separate(lr, 10)

    

