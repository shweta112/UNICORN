from pathlib import Path
import argparse
import yaml
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import DataLoader
import wandb
import pandas as pd

from options import Options
from data_utils import transform_clini_info
from classifier import ClassifierLightning
from utils import get_loss_weighting
from plotting import plot_confusion_matrix, create_report
from data_utils import MILDatasetIndices as Dataset
from data_utils import get_cohort_df, split_dataframe_by_patient

def main(cfg):

    pl.seed_everything(cfg.seed, workers=True)
    base_path = Path(cfg.save_dir) 

    base_path = base_path / cfg.name
    model_path = base_path / "models"
    fold_path = base_path / "folds"
    result_path = base_path / "results"

    base_path.mkdir(parents=True, exist_ok=True)
    fold_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)

    print("\n--- load dataset ---")
    data = get_cohort_df(cfg)
    transform_clini_info(data,cfg,0.02,0.035)

    # --------------------------------------------------------
    # k-fold cross validation
    # --------------------------------------------------------

    model_outputs = pd.DataFrame()
    performance_summaries = pd.DataFrame()

    patient_df = data.groupby("PATIENT").first().reset_index()
    df_splits=split_dataframe_by_patient(patient_df,cfg.folds)

    for i in range(cfg.folds):
        (base_path /"folds"/ f"fold{i}").mkdir(parents=True, exist_ok=True)
        test_fold = df_splits[(i + 4) % 5]
        test_fold.to_csv(base_path /"folds"/ f"fold{i}"/ f"test_df.csv")
    
    for i in range(cfg.folds):
        # training dataset
        training_folds = pd.concat([df_splits[(i + j) % 5] for j in range(3)])
        eval_fold = df_splits[(i + 3) % 5]
        test_fold = df_splits[(i + 4) % 5]
        test_fold.to_csv(result_path / f"fold{i}_test_df.csv")

        train_dataset = Dataset(
            training_folds,
            [cfg.target],
            num_tiles=cfg.num_tiles,
            pad_tiles=cfg.pad_tiles,
            norm=cfg.norm,
            num_classes=cfg.num_classes,
            clini_info=cfg.clini_info
        )

        print(f"num training samples in fold {i}: {len(train_dataset)}")
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.bs,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        # validation dataset
        val_dataset = Dataset(
            eval_fold, [cfg.target], num_classes=cfg.num_classes, norm=cfg.norm,clini_info=cfg.clini_info
        )

        print(f"num validation samples in fold {i}: {len(val_dataset)}")
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        test_dataset = Dataset(
            test_fold,
            [cfg.target],
            num_classes=cfg.num_classes,
            norm=cfg.norm,
            clini_info=cfg.clini_info
        )

        print(f"num test samples in fold {i}: {len(test_dataset)}")

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )


        cfg.loss_weight=get_loss_weighting(np.array((training_folds.TARGET.values),dtype=float))

        model = ClassifierLightning(cfg)

        logger = WandbLogger(
            project=cfg.project,
            config=cfg,
            name=f"{cfg.name}_fold{i}",
            save_dir=cfg.save_dir,
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )

        csv_logger = CSVLogger(
            save_dir=result_path,
            name=f"fold{i}",
        )


        checkpoint_callback = ModelCheckpoint(
            monitor="loss/val",
            dirpath=model_path,
            filename=f"best_model_{cfg.name}_fold{i}",
            save_top_k=1,
            mode="max" if cfg.stop_criterion == "auroc" else "min",
        )
 

        trainer = pl.Trainer(
            logger=[logger, csv_logger],
            precision="16-mixed",
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            gradient_clip_val=1,
            callbacks=[checkpoint_callback],#StochasticWeightAveraging(swa_lrs=4.0e-05)], #
            max_epochs=cfg.num_epochs,
            devices=1,
            accelerator="gpu"
        )

        results_val = trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            ckpt_path=cfg.resume,
        )

        results_test = trainer.test(
            model,
            test_dataloader,
            ckpt_path="best",
        )


        model.outputs.to_csv(result_path / f"fold{i}" / f"outputs.csv")
        model.stain_importance.to_csv( result_path / f"fold{i}" / f"stain_eval.csv")
        model_outputs = pd.concat([model_outputs, model.outputs],ignore_index=True)

        logger.log_table(
            key="results",
            columns=[k for k in results_test[0].keys()],
            data=[[v for v in results_test[0].values()]],
        )

        # Convert the dictionary to a DataFrame with specified index
        results_df = pd.DataFrame(results_test[0],index=[0])

        # Concatenate performance_summaries and results_df
        performance_summaries = pd.concat([performance_summaries, results_df], ignore_index=True)

        wandb.finish()  # required for new wandb run in next fold

    summary_logger = WandbLogger(
        project=cfg.project,
        config=cfg,
        name=f"{cfg.name}_summary",
        save_dir=cfg.save_dir,
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb_run = summary_logger.experiment

    # Log table
    wandb_table = wandb.Table(data=model_outputs)
    wandb_table_summary = wandb.Table(data=performance_summaries)
    wandb_run.log({"results": wandb_table})
    wandb_run.log({"results_summary": wandb_table_summary})

    plot_confusion_matrix(
        model_outputs["ground_truth"].values,
        model_outputs["predictions"].values,
        ["AIT", "PIT", "EFA", "LFA", "CFA"],
        base_path,
        title= 'confusion matrix normal',
        wandb_run=wandb_run
    )
    plot_confusion_matrix(
        model_outputs["ground_truth"].values,
        model_outputs["predictions"].values,
        ["AIT", "PIT", "EFA", "LFA", "CFA"],
        base_path,
        normalize=False,
        wandb_run=wandb_run
    )
    
    # save results to csv file
    model_outputs.to_csv(Path(base_path)/'model_outputs.csv')
    if cfg.visualize:
        create_report(model_outputs["ground_truth"].values,model_outputs["predictions"].values,base_path,wandb_run)
    wandb.finish()

if __name__ == "__main__":
    parser = Options()
    args = parser.parse()

    # Load the configuration from the YAML file
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update the configuration with the values from the argument parser
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != "config_file":
            config[arg_name]["value"] = getattr(args, arg_name)

    print("\n--- load options ---")
    for name, value in sorted(config.items()):
        print(f"{name}: {str(value)}")

    config = argparse.Namespace(**config)
    main(config)
