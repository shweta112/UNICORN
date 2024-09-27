import argparse
from pathlib import Path

import pandas as pd
import tqdm
import yaml
from data_utils import InferenceDataset
from options import Options
from torch import no_grad
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torch
from models.aggregators.model_utils import (normalize_dict_values,
                                            get_networks)

from models.aggregators.unicorn import MultiTransformer as Model
from plotting import attention_visualization_offset
from torch.nn import functional as F
import matplotlib.pyplot as plt

def main(cfg):
    base_path = Path(cfg.save_path)  # adapt to own target path
    
    base_path = base_path / cfg.name

    model_outputs = pd.DataFrame()
    cfg.loss_weight = [1,1,1,1,1]
    model = InferenceLightning(cfg, Path(cfg.model_path) / ("best_model_" + cfg.name + "_fold" + str(cfg.fold) + ".ckpt"))
    model.to('cuda')
    model.on_test_epoch_start()

    test_dataset = InferenceDataset(
        folder=cfg.test_folder
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    with no_grad():
        for batch in tqdm.tqdm(test_dataloader):
            model.test_step(batch, 0)
    
    model_outputs = pd.concat(
        [model_outputs, model.outputs],
        ignore_index=True,
    )
    model_outputs.to_csv(Path(base_path) / "model_outputs_inference_offset.csv")
    # Concatenate performance_summaries and results_df
    # performance_summaries = pd.concat([performance_summaries, results_df], ignore_index=True)



class InferenceLightning(pl.LightningModule):
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.config = config

        if config.feats.startswith('resnet'):
            input_dim = config.input_dim['resnet']
        elif config.feats.startswith('ctranspath'):
            input_dim = config.input_dim['ctranspath']
        elif config.feats.startswith('retccl'):
            input_dim = config.input_dim['retccl']

        self.attention_visualization= attention_visualization_offset
        self.stain_dropout=config.stain_dropout
        self.clini_info_dropout=config.clini_info_dropout

        subnetworks=get_networks(config,config.input_dims)

        self.model = Model(
            num_classes=config.num_classes,
            mlp_dim=512,
            dropout=0.2,
            num_base_networks=len(config.cohort_data[config.cohort]['slide_csv'])+len(config.clini_info),
            stain_dropout=self.stain_dropout,
            clini_info_dropout=self.clini_info_dropout,
            subnetworks=subnetworks)

        self.stain_importance=pd.DataFrame()
        self.criterion=None
        self.load_state_dict(torch.load(checkpoint)['state_dict'],strict=False)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def on_test_epoch_start(self) -> None:
        # save test outputs in dataframe per test dataset
        column_names = ["patient","staining_permutation", "ground_truth", "predictions", "logits", "correct","staining_contributions"]
        self.outputs = pd.DataFrame(columns=column_names)
        self.model.stain_dropout=0
        self.model.clini_info_dropout=0
        self.model.eval()
        self.model.test_mode=True


    def test_step(self, batch, batch_idx):

        x, coords, _,patient, _ ,filenames = batch  # x = features, y = labels
        x = [t.to('cuda') for t in x]  # Move x tensor to GPU
        logits, features = self.forward(x)

        probs=F.softmax(logits,dim=1)            
        preds = torch.argmax(probs, dim=1, keepdim=True)
  
        probs = probs.unsqueeze(-1)
        attentions,staining_contributions = self.model.attention_rollout(x)
        
        outputs = pd.DataFrame(
            data=[
                [
                    Path(filenames[0][0]).stem,
                    [index for index, value in enumerate(x) if value is not None],
                    preds.cpu().item(),
                    logits.cpu().numpy(),
                    staining_contributions.tolist(),
                    features.cpu().numpy()
                ]
            ],
            columns=[
                "patient",
                "staining_permutation",
                "predictions",
                "logits",
                'staining_contributions',
                'features'
            ],
        
        )
        self.outputs = pd.concat([self.outputs, outputs], ignore_index=True)


        staining_contrib_dict={}

        for filename,staining_contribution in zip(filenames,staining_contributions):
            staining_contrib_dict[filename[0].split("_")[-1].replace(".h5","")]=staining_contribution

        staining_contrib_dict=normalize_dict_values(staining_contrib_dict)
        patchwise_prediction=self.model.predict_single_patches(x)
        class_attention,patchwise_class_prob_of_prediction,attentions_normalized=self.model.get_class_attention(attentions,patchwise_prediction,preds)
        self.attention_visualization(class_attention,batch, self.config,"overlap_class_attention_")
        self.attention_visualization(patchwise_prediction,batch,self.config,"overlap_multiclass_",class_visualization=True)
        self.attention_visualization(patchwise_class_prob_of_prediction,batch,self.config,"overlap_class_",plt.cm.inferno)
        self.attention_visualization(attentions_normalized,batch,self.config,"overlap_attention_")


        
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