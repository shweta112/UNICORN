import argparse
import itertools
from pathlib import Path

import pandas as pd
import tqdm
import yaml
from classifier import ClassifierLightning
from data_utils import MILDatasetIndices, get_cohort_df, transform_clini_info
from options import Options
from torch import no_grad
from torch.utils.data import DataLoader

def generate_combinations():
    numbers = [0, 1, 2, 3]
    combinations = []

    for r in range(1, 5):
        for combo in itertools.combinations(numbers, r):
            combinations.append(combo)

    return combinations


def process_batch(batch, keep_indices,device):
    x, a, b, c, d, e = batch  # x = features, y = labels

    # Replace elements not in keep_indices with None
    x_filtered = [None] * len(x)

    for i in range(len(x)):
        if i in keep_indices and len(x[i].shape)==3:
            x_filtered[i] = x[i].to(device)
    # Create a copy of x to preserve the original values
    return x_filtered, a, b, c, d, e


def main(cfg):
    base_path = Path(cfg.save_dir)  # adapt to own target path
    cfg.name = f"{cfg.name}"
    cfg.visualize = False
    base_path = base_path / cfg.name
    model_path = base_path / "models"
    fold_path = base_path / "folds"

    data = get_cohort_df(cfg)
    transform_clini_info(data,cfg,0.02,0.035)
    all_staining_permutations = generate_combinations()
    model_outputs = pd.DataFrame()

    for l in range(cfg.folds):

        test_df = pd.read_csv(fold_path/f"fold{l}/test_df.csv", index_col="Unnamed: 0")
        cfg.loss_weight = [1,1,1,1,1]

        model = ClassifierLightning(
            cfg, model_path / ("best_model_" + cfg.name + "_fold" + str(l) + ".ckpt")
        )
        model.to('cuda')
        model.eval()
        model.on_test_epoch_start()
        model.model.test_mode=True

        test_dataset = MILDatasetIndices(
            test_df,
            [cfg.target],
            num_classes=cfg.num_classes,
            clini_info=cfg.clini_info,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        with no_grad():
            for batch in tqdm.tqdm(test_dataloader):
                for staining_indices in all_staining_permutations:
                    batch_processed = process_batch(batch, staining_indices,model.device)
                    if not all(item is None for item in batch_processed[0]):
                        model.test_step(batch_processed, 0)
        model_outputs = pd.concat(
            [model_outputs, model.outputs],
            ignore_index=True,
        )
    model_outputs.to_csv(Path(base_path) / "model_outputs_inference_075.csv")
    # Concatenate performance_summaries and results_df
    # performance_summaries = pd.concat([performance_summaries, results_df], ignore_index=True)


if __name__ == "__main__":
    parser = Options()
    args = parser.parse()
    args.config_file="/baseline_dropout_0.75.yaml"
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


# for stain_feature,filename,staining_contribution in zip(x,filenames,staining_contributions):
#             logits_stain = self.forward(stain_feature.unsqueeze(0))
#             loss_stain = self.criterion(logits_stain, y1)

#             probs_stain = F.softmax(logits_stain, dim=1)
#             preds_stain = torch.argmax(probs_stain, dim=1, keepdim=True)
#             data = {
#             'Probs': [probs_stain.cpu().numpy().tolist()],
#             'Preds': [preds_stain.item()],
#             'Loss': [loss_stain.item()],
#             'Attention':[staining_contribution],
#             'Pred_all':[preds.item()],
#             'Probs_all':[probs.cpu().numpy().tolist()],
#             'gt': y.item()
#             }

#             new_entry = pd.DataFrame(data, index=[filename])
#             self.stain_importance = pd.concat([self.stain_importance, new_entry])
