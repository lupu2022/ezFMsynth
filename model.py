import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import math
import msgpack
import yaml
import sys

def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)

def upsample(signal, block_size):
    signal = signal.permute(0, 2, 1)

    signal = torch.nn.Upsample(scale_factor = block_size)(signal)

    signal = signal.permute(0, 2, 1)
    return signal

def smooth_upsample(signal, block_size):
    # just do simple upsampling
    signal = signal.permute(0, 2, 1)
    signal = signal.unsqueeze(0)

    signal = nn.functional.interpolate(signal, size=[signal.shape[2], signal.shape[-1] * block_size], mode='bilinear')

    signal = signal.squeeze(0)
    signal = signal.permute(0, 2, 1)
    return signal

def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]
    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win
    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    return amp

def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output

def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,                     ## input
            s,                          ## n_fft
            int(s * (1 - overlap)),     ## hop_length
            s,                          ## win_length
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts

def safe_log(x):
    return torch.log(x + 1e-7)

def multiscale_loss(s, y, scales, overlap, weights):
    ori_stft = multiscale_fft(s, scales, overlap)
    rec_stft = multiscale_fft(y, scales, overlap)

    loss = 0
    for i in range(0, len(ori_stft)):
        s_x = ori_stft[i];
        s_y = rec_stft[i];
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + (lin_loss + log_loss) * weights[i]

    return loss

class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x

class DDSP(nn.Module):
    def __init__(self, input_length, input_hop, hidden_size, FM_sync, FM_config, FM_amp, noise_bands, sampling_rate, block_size, reverb_length):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.block_size = block_size * input_hop
        self.input_length = input_length
        self.input_hop = input_hop
        self.FM_sync = FM_sync
        self.FM_config = FM_config
        self.FM_amp = FM_amp
        self.noise_bands = noise_bands

        # pitch & loudness are inputs
        self.pitch_mlp = mlp(input_length, hidden_size, 2)
        self.loudness_mlp = mlp(input_length, hidden_size, 2)

        # combing pitch & loudness
        self.out_mlp = mlp(hidden_size * 2, hidden_size, 3)

        # finaly output for FM synthesizer
        index_number = len(self.FM_config)
        self.alpha_proj = nn.Linear(hidden_size, index_number);
        self.beta_proj = nn.Linear(hidden_size, index_number);
        self.bands_proj = nn.Linear(hidden_size, self.noise_bands)

        # long FIR kernel for post-processing
        self.reverb = Reverb(reverb_length, sampling_rate)

        # internal help member
        self.timeline = None

    def forward(self, pitch_ext, loudness_ext):
        pitch = torch.log2( (pitch_ext[:, 0:-1:self.input_hop, :] + 0.1) / 440.0 ) * 12;
        loudness = loudness_ext[:, 0:-1:self.input_hop, :]

        hidden_in = torch.cat([
            self.pitch_mlp(pitch),
            self.loudness_mlp(loudness)
        ], -1)

        # fetch current target pitch&loudness
        hidden_in = torch.tanh(hidden_in)
        hidden_out = self.out_mlp(hidden_in)

        ################# building indexs fucntion #############
        if ( self.timeline == None ):
            t = torch.arange(self.block_size, device = pitch.device) / self.block_size
            t = t.reshape( [1, self.block_size, 1] );
            self.timeline = t.repeat(pitch.size()[0], pitch.size()[1],  len(self.FM_config))

        alpha = self.FM_amp * torch.sigmoid(self.alpha_proj(hidden_out)) + 1e-7;
        beta = self.FM_amp * torch.sigmoid(self.beta_proj(hidden_out)) + 1e-7;
        alpha = upsample(alpha, self.block_size)
        beta = upsample(beta, self.block_size)

        fms = self.timeline * alpha + (1.0 - self.timeline) * beta

        ################# harmonic synthsis ####################
        # 1. upsample pitch to block_size
        pitch = pitch_ext[:, :, -1:]        # keep current target pitch
        pitch = upsample(pitch, self.block_size // self.input_hop)

        # 2. FM synthesizer
        omegas = []
        omega = torch.cumsum(2 * math.pi * pitch  / self.sampling_rate, 1)
        for i in range(0, len(self.FM_config)):
            omegas.append( omega * (self.FM_config[i]))

        '''
        Flute FM Synth - with phase wrapping (it does not change behaviour)
        PATCH NAME: FLUTE 1
            1.5->2->|
              2->1->|
                 1->|->1->out
        '''
        if self.FM_sync == "FLUTE":
            op6_out = fms[..., 0:1] * torch.sin(omegas[0])
            op5_out = fms[..., 1:2] * torch.sin(omegas[1] + op6_out)

            op4_out = fms[..., 2:3] * torch.sin(omegas[2])
            op3_out = fms[..., 3:4] * torch.sin(omegas[3] + op4_out)

            op2_out = fms[..., 4:5] * torch.sin(omegas[4])

            op1_out = fms[..., 5:6] * torch.sin(omegas[5] + op5_out + op3_out + op2_out)


        harmonic = op1_out.squeeze(-1)

        ################# noise (resudual) filter ####################
        bands = 2.0 * torch.sigmoid(self.bands_proj(hidden_out) - 5.0) + 1e-7
        impulse = amp_to_impulse_response(bands, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2.0 - 1.0

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1)

        ################# reverb (long FIR) filter ####################
        reverb = self.reverb(harmonic + noise).squeeze(-1)

        return harmonic, reverb

    def verify(self, pitch, loudness):
        hidden_in = torch.cat([
            self.pitch_mlp(pitch),
            self.loudness_mlp(loudness)
        ], -1)

        # fetch current target pitch&loudness
        hidden_in = torch.tanh(hidden_in)
        hidden_out = self.out_mlp(hidden_in)

        alpha = torch.sigmoid(self.alpha_proj(hidden_out));
        beta = torch.sigmoid(self.beta_proj(hidden_out));
        bands = torch.sigmoid(self.bands_proj(hidden_out) - 5.0)

        return alpha, beta, bands


def dump(config, weight_file):
    model = DDSP(**config)
    result = model.load_state_dict( torch.load(weight_file, map_location=torch.device('cpu')), False)
    assert( len(result.missing_keys) == 0)

    model.eval()
    sdict = model.state_dict()

    allWeights = []
    for k in sdict.keys():
        t = sdict[k]

        if k.startswith("reverb"):
            continue

        print(k, t.shape)

        if ( len(t.shape) > 0):
            ## just pass by saved to mskpack
            vlist = t.numpy().flatten().tolist()
            weight = [k, "float", t.shape, vlist]
            allWeights.append(weight)
    d =  msgpack.packb(allWeights, use_bin_type=True)
    with open('ddsp.msg', "wb") as outfile:
        outfile.write(d)

    ## dump reverb impulse
    impulse = model.reverb.build_impulse().reshape(-1).detach().numpy()
    d = msgpack.packb(impulse.tolist(), use_bin_type=True)
    with open("reverb.msg", "wb") as outfile:
        outfile.write(d)

    return allWeights

def verify(config, weight_file):
    model = DDSP(**config)
    result = model.load_state_dict( torch.load(weight_file, map_location=torch.device('cpu')), False)

    model.eval()

    p = torch.zeros(1, 1, config["input_length"])
    p.fill_(3.14)
    l = torch.zeros(1, 1, config["input_length"])
    l.fill_(0.314)

    return  model.verify(p, l)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please input config_file and weight_file");
        sys.exit(0);

    with open(sys.argv[1], "r") as data:
        config = yaml.safe_load(data)

    config["model"].pop("mean_loudness")
    config["model"].pop("std_loudness")

    wfile = sys.argv[2]

    dump(config["model"], wfile)
    debug = verify(config["model"], wfile)
