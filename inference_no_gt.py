import argparse
from pathlib import Path
import copy

import pandas as pd
import tqdm
import yaml
from classifier import ClassifierLightning
from data_utils import InferenceDataset
from options import Options
from torch import no_grad
from torch.utils.data import DataLoader

def process_batch(batch, keep_indices):
    x, a, b, c, d, e = batch  # x = features, y = labels

    # Replace elements not in keep_indices with None
    x_filtered = [None] * len(x)

    for i in range(len(x)):
        if i in keep_indices:
            x_filtered[i] = x[i]
    # Create a copy of x to preserve the original values
    return x_filtered, a, b, c, d, e


def main(cfg):

    model_outputs = pd.DataFrame()
    cfg.loss_weight = [1,1,1,1,1]
    model = ClassifierLightning(cfg, Path(cfg.model_path) / ("best_model_" + cfg.name + "_fold" + str(cfg.fold) + ".ckpt"))
    model.to('cuda')
    model.eval()
    model.model.test_mode=True
    model.on_test_epoch_start()


    test_dataset = InferenceDataset(
        cfg.test_folder
    )


    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    with no_grad():
        # for batch in tqdm.tqdm(test_dataloader):
        #     for i in range(4):
        #         batch_copy=copy.deepcopy(batch)
        #         processed_input=batch_copy[0]
        #         processed_input=[None]*4

        #         if len(batch_copy[0][i].shape)==3:
        #             processed_input[i]=batch_copy[0][i]

        #         if not all(item is None for item in processed_input):      
        #             batch_copy[0]=processed_input
        #             model.test_step(batch_copy, 0)
    # with no_grad():
        for batch in tqdm.tqdm(test_dataloader):

            batch_copy=copy.deepcopy(batch)
            processed_input=batch_copy[0]
                
            for j in range(4):
                if len(processed_input[j].shape)<=2:
                    pass

            if not all(item is None for item in processed_input):      
                batch_copy[0]=processed_input
                try:
                    model.test_step(batch_copy, 0)
                except Exception as e:
                    print(e)
                
    model_outputs = pd.concat(
        [model_outputs, model.outputs],
        ignore_index=True,
    )
    save_path=Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model_outputs.to_csv(save_path / "model_outputs_inference_21_all.csv")
    # Concatenate performance_summaries and results_df
    # performance_summaries = pd.concat([performance_summaries, results_df], ignore_index=True)


if __name__ == "__main__":
    parser = Options()
    args = parser.parse()
    args.config_file='/inference_config.yaml'
    # Load the configuration from the YAML file
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update the configuration with the values from the argument parser
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != "config_file":
            config[arg_name]["value"] = getattr(args, arg_name)

    # Create a flat config file without descriptions
    # config = {k: v['value'] for k, v in config.items()}

    print("\n--- load options ---")
    for name, value in sorted(config.items()):
        print(f"{name}: {str(value)}")

    config = argparse.Namespace(**config)
    main(config)