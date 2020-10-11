
class SignalAugmentation:
    def __init__(self, progress=None):
        self.functions = []
        if progress is None:
            self.progress = lambda x, **kwargs: x
        else:
            self.progress = progress

    def apply(self, xs):
        for f in self.progress(self.functions):
            xs = f(xs)
        return xs

    def standardize_shape(self, shape, mode='reflect'):
        def _standardize_shape(xs):
            ys = []
            for x in xs:
                sub_shapes = (np.array(shape) - np.array(x.shape)) // 2
                pad_shape = [
                    (ss, ss + int(input_s - ss * 2 - x_shape)) 
                    for ss, x_shape, input_s in zip(sub_shapes, x.shape, shape)
                ]
                
                ys.append(
                    np.pad(x, pad_shape, mode=mode.lower()))
            return ys
        self.functions.append(_standardize_shape)
        return self

    def map(self, func, **kwargs):
        def _map(xs):
            return [func(x, **kwargs) for x in xs]
        self.functions.append(_map)
        return self

    def stft(self, n_sampling=256, hopsize=128, window='hann', pad='constant', log=False):
        """ Short-Time Fourier Transform
        [Hop size, Frequency, Magnitude and Phase] 
        """
        def _stft(xs):
            ys = []
            for x in xs:
                F = librosa.stft(x, n_sampling, hop_length=hopsize, window=window, pad_mode=pad)
                mag = np.abs(F)[..., np.newaxis]
                phase = np.angle(F)[..., np.newaxis]
                if log:
                    mag = np.log(mag)
                ys.append(np.concatenate([mag, phase], axis=2))
            return ys
        self.functions.append(_stft)
        return self
            
    def istft(self, sampling=512, hopsize=256, window=np.hamming, output_shape=None, log=False):
        def _istft(xs):
            ys = []
            for x in xs:
                mag =  x[:, :, 0]
                if log:
                    mag = np.exp(mag)
                F = mag * np.exp(np.complex(0, 1) * x[:, :, 1])
                ys.append(librosa.istft(F, hop_length=hopsize))
            return ys
        self.functions.append(_istft)
        return self

    def mel_spectrum(self, sampling_rate):
        def _mel_spectrum(xs):
            ys = []
            for x in xs:
                x[..., 0] = librosa.feature.melspectrogram(
                    S=x[..., 0] ** 2, sr=sampling_rate, n_mels=x.shape[0],
                    fmin=0.0, fmax=sampling_rate / 2
                )
                ys.append(x)
            return ys
                
        self.functions.append(_mel_spectrum)
        return self

    def inverse_mel_spectrum(self, sampling_rate):
        def _inverse_mell_spectrum(xs):
            ys = []
            for x in xs:
                x[..., 0] = librosa.feature.inverse.mel_to_stft(
                    x[..., 0], sr=sampling_rate, n_fft=(x.shape[0] - 1) * 2,
                    fmin=0.0, fmax=sampling_rate / 2
                )
                ys.append(x)
            return ys
        self.functions.append(_inverse_mell_spectrum)
        return self

    def slice(self, *keys):
        def _slice(xs):
            return [x[keys] for x in xs]
        self.functions.append(_slice)
        return self