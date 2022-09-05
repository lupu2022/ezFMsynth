import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from effortless_config import Config
from os import path
from tqdm import tqdm
import soundfile as sf
from einops import rearrange
import numpy as np

from build_dataset import Dataset
from model import DDSP, multiscale_fft, multiscale_loss

@torch.no_grad()
def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for _, _, l in dataset:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std

def get_scheduler(len_dataset, start_lr, stop_lr, length):
    def schedule(epoch):
        step = epoch * len_dataset
        if step < length:
            t = step / length
            return start_lr * (1 - t) + stop_lr * t
        else:
            return stop_lr

    return schedule

class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    STEPS = 10000
    BATCH = 32
    LR = 1e-3

def do_train():
    args.parse_args()

    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDSP(**config["model"]).to(device)
    #model.load_state_dict( torch.load('./state_000038.pth'), True);

    dataset = Dataset(config["preprocess"]["out_dir"], config["model"]["input_length"], True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        args.BATCH,
        True,
        drop_last=True,
    )

    mean_loudness, std_loudness = mean_std_loudness(dataloader)
    config["model"]["mean_loudness"] = mean_loudness
    config["model"]["std_loudness"] = std_loudness

    print("loudness mean/std = ", mean_loudness, std_loudness)

    writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

    with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
        yaml.safe_dump(config, out_config)

    opt = torch.optim.Adam(model.parameters(), lr=args.LR)

    best_loss = float("inf")
    mean_loss = 0
    n_element = 0
    step = 0
    epochs = int(np.ceil(args.STEPS / len(dataloader)))

    for e in tqdm(range(epochs)):
        for s, p, l in dataloader:
            s = s.to(device)
            p = p.to(device)
            l = l.to(device)
            l = (l - mean_loudness) / std_loudness

            l = l + (torch.rand_like(l) - 0.5) * 0.02;
            y, r = model(p, l)

            if ( e < epochs / 3):
                loss_y = multiscale_loss(s, y, config["train"]["scales"], config["train"]["overlap"], config["train"]["weights"])
                loss_r = multiscale_loss(s, r, config["train"]["scales"], config["train"]["overlap"], config["train"]["weights"])

                loss = 0.5 * loss_y + 0.5 * loss_r
            else:
                loss_r = multiscale_loss(s, r, config["train"]["scales"], config["train"]["overlap"], config["train"]["weights"])
                loss = loss_r

            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("loss", loss.item(), step)

            step += 1

            n_element += 1
            mean_loss += (loss_r.item() - mean_loss) / n_element

            print("==============>", step, mean_loss, loss_r.item(), loss.item() )

        if not e % 2:
            writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
            writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)

            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(
                    model.state_dict(),
                    path.join(args.ROOT, args.NAME, f"state_{e:06d}.pth"),
                )

            mean_loss = 0
            n_element = 0

            audio = torch.cat([s, y], -1).reshape(-1).detach().cpu().numpy()
            sf.write(
                path.join(args.ROOT, args.NAME, f"eval_{e:06d}.wav"),
                audio,
                config["preprocess"]["sampling_rate"],
            )

            audio = torch.cat([s, r], -1).reshape(-1).detach().cpu().numpy()
            sf.write(
                path.join(args.ROOT, args.NAME, f"eval_{e:06d}_.wav"),
                audio,
                config["preprocess"]["sampling_rate"],
            )

if __name__ == "__main__":
    do_train()
