import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from effortless_config import Config
from os import path
from tqdm import tqdm
import soundfile as sf
from einops import rearrange
import numpy as np
import pathlib
import sys

from build_dataset import Dataset
from model import DDSP, multiscale_fft, multiscale_loss

def get_files(data_location, extension):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))

def do_cherry_pick( model_dir ):

    config_file = path.join( model_dir, "config.yaml")

    with open(config_file, "r") as config:
        config = yaml.safe_load(config)

    mean_loudness = config["model"]["mean_loudness"]
    std_loudness = config["model"]["std_loudness"]
    config["model"].pop("mean_loudness")
    config["model"].pop("std_loudness")

    dataset = Dataset(config["preprocess"]["out_dir"], config["model"]["input_length"], False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        16,
        False,
        drop_last=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DDSP(**config["model"]).to(device)

    best_loss = float("inf")
    best_model = ""

    files = get_files(model_dir, "pth")

    for pth_file in files:
        print(">>> model file :" , pth_file )

        model.load_state_dict( torch.load( pth_file, map_location=device  ), True);
        model.eval();

        mean_loss = 0.0;
        n_element = 0
        for s, p, l in dataloader:
            s = s.to(device)
            p = p.to(device)
            l = l.to(device)
            l = (l - mean_loudness) / std_loudness

            y, r = model(p, l)

            loss_r = multiscale_loss(s, r, config["train"]["scales"], config["train"]["overlap"], config["train"]["weights"])

            n_element += 1
            mean_loss += (loss_r.item() - mean_loss) / n_element

            print("==============>", mean_loss, loss_r.item() )

        if ( mean_loss < best_loss ):
            best_loss = mean_loss
            best_model = pth_file;

            audio = r.reshape(-1).detach().cpu().numpy()
            sf.write(
                "cherry.wav",
                audio,
                config["preprocess"]["sampling_rate"],
            )


    print("best model: ", best_loss, best_model )

if __name__ == "__main__":
    do_cherry_pick( sys.argv[1] )
