import torch
import os
import logging
from argparse import ArgumentParser
from src.utils.dataloaders import load_dataloader
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4
from src.utils.parameters import write_params_to_file, load_parameters, instanciate_cls
from src.experiment.base import AbstractExperiment

if __name__ == "__main__":

    logging_message = "[AROB-2025-KAPTIOS]"
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser()
    parser.add_argument("--params", type=str, default='./params/mnist.yaml')
    args = parser.parse_args()
    params = load_parameters(args.params)

    data_params = params['dataset']['parameters']
    model_params = params['model']['parameters']
    xp_params = params['experiment']['parameters']

    ## ========== INIT ========== ##

    gpu = torch.cuda.is_available()
    DEVICE = torch.device("cuda") if gpu else torch.device("cpu")

    if gpu:
        torch.cuda.manual_seed_all(xp_params["seed"])
    else:
        torch.manual_seed(xp_params["seed"])

    experiment_id = str(uuid4())[:8]
    experiment_name = f"{params['dataset']['name']}_{params['model']['name']}_{experiment_id}"
    logging.info(
        'Initialization of the experiment protocol - {}'.format(experiment_name))
    log_dir = os.path.join(xp_params["log_dir"], experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    write_params_to_file(params, log_dir)
    log_interval = max(50 // data_params["batch_size"], 1)

    # ========== DATALOADER ========== ##

    train_dl, test_dl = load_dataloader(
        params['dataset']['name'],
        data_params,
        gpu)

    logging.info('Dataloaders successfully loaded.')

    ## ========== MODEL ========== ##

    model = instanciate_cls(
        params['model']['module_name'], params['model']['name'], model_params)
    logging.info(f"Model - {model}")
    model.to(DEVICE)

    # ========== TRAINING ========== ##

    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

    logging.info(f'Running on device : {DEVICE}')
    experiment: AbstractExperiment = instanciate_cls(
        params['experiment']['module_name'],
        params['experiment']['name'],
        {
            "model": model,
            "writer": writer,
            "log_interval": log_interval,
            "lr": xp_params['lr'],
        }
    )

    experiment.fit(train_dl, test_dl, xp_params['num_epochs'])
