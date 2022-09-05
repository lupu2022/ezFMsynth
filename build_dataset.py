import yaml
import pathlib
import librosa as li
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
from scipy.io import wavfile
import pyloudnorm as pyln
import crepe
import torch

def extract_loudness2(signal, sampling_rate, hop_size, internal_block_size = 0.3):
    meter = pyln.Meter(sampling_rate, block_size=internal_block_size)
    block_size = int(internal_block_size * sampling_rate)

    out_number = signal.shape[0] // hop_size

    pad = block_size//2 + 1
    signal = np.pad( signal, (pad, pad) )

    loudness_vals = np.zeros( out_number, dtype="float" )

    for i in range(0, out_number):
        begin = i * hop_size
        end = begin + block_size
        s = signal[begin:end]

        l = meter.integrated_loudness(s)
        loudness_vals[i] = l

    loudness_vals[ np.isinf(loudness_vals) ] = -70.0

    return loudness_vals

def extract_pitch(signal, sampling_rate, block_size):
    length = signal.shape[-1] // block_size
    f0 = crepe.predict(
        signal,
        sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    conf = f0[2].reshape(-1)[:-1]
    f0 = f0[1].reshape(-1)[:-1]

    conf[ np.isnan(conf) ] = 1.0
    f0[ np.isnan(conf) ] = 0.0

    ## conver f0 to MIDI
    #key = np.round(np.log2( (f0 + 0.01) / 440.0) * 12 + 69)
    #f0 = 440 * np.power(2, (key - 69) / 12);

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )
        conf = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, conf.shape[-1], endpoint=False),
            conf,
        )

    return f0, conf

def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))

dropped = 0
def preprocess(f, sampling_rate, block_size, signal_length, conf_threshold, **kwargs):
    x, sr = li.load(f, sr=sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    pitch, conf = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness2(x, sampling_rate, block_size)

    hop_size = int( signal_length )
    hop_num = int( x.shape[0] / hop_size)

    assert( sr == sampling_rate)
    assert( int(hop_size/block_size) * block_size == hop_size)

    xx = []
    pp = []
    ll = []
    for h in range(0, hop_num):
        left = h * hop_size
        right = left + signal_length

        bleft = int(left / block_size)
        bright = int(right / block_size)

        mean_conf = np.mean(conf[bleft:bright])
        if (mean_conf >= conf_threshold):
            xx.append( x[left:right] )
            pp.append( pitch[bleft:bright] )
            ll.append( loudness[bleft:bright] )
        else:
            global dropped
            dropped = dropped + 1

    if len(xx) > 0:
        x = np.concatenate(xx, 0)
        pitch = np.concatenate(pp, 0)
        loudness = np.concatenate(ll, 0)

        x = x.reshape(-1, signal_length)
        pitch = pitch.reshape(x.shape[0], -1)
        loudness = loudness.reshape(x.shape[0], -1)

        return x, pitch, loudness
    else:
        return [], [], []


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir, input_length, is_train):
        super().__init__()
        self.input_length = input_length
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))

        self.is_train = is_train;
        self.split = int(self.signals.shape[0] * 0.10);

    def __len__(self):
        if ( self.is_train ) :
            return self.signals.shape[0] - self.split
        return self.split;

    def __getitem__(self, idx):
        if ( self.is_train ) :
            idx = idx + self.split

        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])

        pitch = torch.zeros([p.shape[0], self.input_length]).to(p)
        loudness = torch.zeros([l.shape[0], self.input_length]).to(l)

        extend = self.input_length - 1
        p = torch.nn.functional.pad(p, [extend, 0])
        l = torch.nn.functional.pad(l, [extend, 0])

        p[:extend+1] = p[extend]
        l[:extend+1] = l[extend]

        block_num = pitch.shape[0]
        for i in range(0, block_num):
            pitch[i, :] = p[i:i+self.input_length]
            loudness[i, :] = l[i:i+self.input_length]

        return s, pitch, loudness

def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_files(**config["preprocess"])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []

    num = 0
    for f in pb:
        pb.set_description(str(f))
        x, p, l = preprocess(f, **config["preprocess"])
        if len(x) > 0:
            num = num + x.shape[0]
            signals.append(x)
            pitchs.append(p)
            loudness.append(l)

            print("Current sample number = %d, dropped = " % num, dropped)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)

'''
class args(Config):
    CONFIG = "config.yaml"

args.parse_args()
with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

dataset = Dataset(config["preprocess"]["out_dir"], config["preprocess"]["input_length"], True)
s, p, l = dataset[0]
'''

if __name__ == "__main__":
    main()
