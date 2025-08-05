from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from models.aggregators.model_utils import (normalize_dict_values,
                                            get_networks)
from torch.nn import functional as F
from utils import get_loss, get_optimizer
import matplotlib.pyplot as plt
from plotting import attention_visualization, create_report_card
from models.aggregators.attentionmil import AttentionMIL
from models.aggregators.unicorn import MultiTransformer 
from models.aggregators.perceiver import Perceiver
from models.aggregators.transformer import Transformer

class ClassifierLightning(pl.LightningModule):
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.config = config

        self.create_report_card=create_report_card
        self.attention_visualization=attention_visualization

        self.stain_dropout=config.stain_dropout
        self.clini_info_dropout=config.clini_info_dropout

        if config.model=="unicorn":
            subnetworks=get_networks(config,config.input_dims)

            self.model = MultiTransformer(
                num_classes=config.num_classes,
                mlp_dim=512,
                dropout=0.2,
                num_base_networks=len(config.cohort_data[config.cohort]['slide_csv'])+len(config.clini_info),
                stain_dropout=self.stain_dropout,
                clini_info_dropout=self.clini_info_dropout,
                subnetworks=subnetworks,
                register=config.register)
            

        elif config.model=="AttentionMIL":
                self.model = AttentionMIL(
                num_classes=config.num_classes,
                num_features=config.input_dims[0],
                mlp_dim=256,
                stain_dropout=self.stain_dropout,
                clini_info=config.clini_info)

        elif config.model=="Transformer":
            settings=config.subnetwork["transformer"]
            self.model = Transformer(**settings,num_classes=config.num_classes,input_dim=config.input_dims[0],subnetwork=False)

        elif config.model=="Perceiver":
            settings=config.subnetwork["perceiver"]
            self.model = Perceiver(**settings,num_classes=config.num_classes,subnetwork=False)




        self.stain_importance=pd.DataFrame()
        self.lr = config.lr
        self.wd = config.wd

        self.acc_train = torchmetrics.Accuracy(
            task=config.task, num_classes=config.num_classes
        )
        self.acc_val = torchmetrics.Accuracy(
            task=config.task, num_classes=config.num_classes
        )
        self.acc_test = torchmetrics.Accuracy(
            task=config.task, num_classes=config.num_classes
        )

        self.auroc_val = torchmetrics.AUROC(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.auroc_test = torchmetrics.AUROC(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.f1_val = torchmetrics.F1Score(
            task=config.task,
            num_classes=config.num_classes,
            average='macro'
        )
        self.f1_test = torchmetrics.F1Score(
            task=config.task,
            num_classes=config.num_classes,
            average='macro'
        )

        self.precision_val = torchmetrics.Precision(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.precision_test = torchmetrics.Precision(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.recall_val = torchmetrics.Recall(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.recall_test = torchmetrics.Recall(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.specificity_val = torchmetrics.Specificity(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.specificity_test = torchmetrics.Specificity(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.criterion = get_loss(
            config.criterion,weight=torch.Tensor(config.loss_weight))
        self.num_classes=config.num_classes
        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint)['state_dict'],strict=True)

        

    def forward(self, x):
        logits = self.model(x)
        return logits

    def configure_optimizers(self):
        optimizer = get_optimizer(
            name=self.config.optimizer, model=self.model, lr=self.lr, wd=self.wd
        )
        # TODO add lr scheduler
        # scheduler = self.scheduler(
        #     name=self.config.scheduler,
        #     optimizer=optimizer,
        # )
        return [optimizer]  # , [scheduler]

    def training_step(self, batch, batch_idx):
        x, _, y, _, _, _ = batch  # x = features, y = labels
        logits,_ = self.forward(x)
        loss = self.criterion(logits, y)

        y = torch.argmax(y, dim=1, keepdim=True)
        probs=F.softmax(logits,dim=1)
        # self.lr_schedulers().step()

        probs = probs.unsqueeze(-1)
        self.acc_train(probs, y)
        self.log(
            "acc/train", self.acc_train, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y, _, _, _ = batch  # x = features, y = labels
        logits,_ = self.forward(x)
        loss = self.criterion(logits, y)

        probs=F.softmax(logits,dim=1)
        y = torch.argmax(y, dim=1, keepdim=True)

        probs = probs.unsqueeze(-1)
        self.acc_val(probs, y)
        self.auroc_val(probs, y)
        self.f1_val(probs, y)
        self.precision_val(probs, y)
        self.recall_val(probs, y)
        self.specificity_val(probs, y)

        self.log("loss/val", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("acc/val", self.acc_val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "auroc/val", self.auroc_val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log("f1/val", self.f1_val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "precision/val",
            self.precision_val,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "recall/val", self.recall_val, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "specificity/val",
            self.specificity_val,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

    def on_validation_epoch_start(self) -> None:
        self.model.stain_dropout=0
        self.model.clini_info_dropout=0

    def on_train_epoch_start(self) -> None:
        self.model.stain_dropout=self.stain_dropout
        self.model.clini_info_dropout=self.clini_info_dropout

    def on_test_epoch_start(self) -> None:
        # save test outputs in dataframe per test dataset
        column_names = ["patient","staining_permutation", "ground_truth", "predictions", "logits", "correct","staining_contributions"]
        self.outputs = pd.DataFrame(columns=column_names)
        self.model.stain_dropout=0
        self.model.clini_info_dropout=0


    def test_step(self, batch, batch_idx):
        
        x, _, y1, patient, feature_args, filenames = batch  # x = features, y = labels
        x = [t.to('cuda') if t is not None else None for t in x]
        y1=y1.to('cuda')
        logits, features = self.forward(x)
        loss = self.criterion(logits, y1)

        # if  self.num_classes>2:
        probs=F.softmax(logits,dim=1)
        preds = torch.argmax(probs, dim=1, keepdim=True)
        y = torch.argmax(y1, dim=1, keepdim=True)
        probs = probs.unsqueeze(-1)
        
        # elif  self.num_classes==2:
        #     probs=F.sigmoid(logits)
        #     y=y1
        #     print(probs.size())
        #     preds=int((probs>0.5).item())
            #probs = probs.unsqueeze(-1)

        self.acc_test(probs, y)

        self.auroc_test(probs, y)
        self.f1_test(probs, y)
        self.precision_test(probs, y)
        self.recall_test(probs, y)
        self.specificity_test(probs, y)

        self.log("loss/test", loss, prog_bar=False)
        self.log("acc/test", self.acc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "auroc/test", self.auroc_test, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log("f1/test", self.f1_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "precision/test",
            self.precision_test,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "recall/test",
            self.recall_test,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "specificity/test",
            self.specificity_test,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )


        attentions,staining_contributions = self.model.attention_rollout(x)
        
        outputs = pd.DataFrame(
            data=[
                [
                    patient[0],
                    [index for index, value in enumerate(x) if value is not None],
                    y.item(),
                    preds.cpu().numpy(),
                    logits.cpu().numpy(),
                    (y == preds).int().item(),
                    staining_contributions.tolist(),
                    features.cpu().numpy()
                ]
            ],
            columns=[
                "patient",
                "staining_permutation",
                "ground_truth",
                "predictions",
                "logits",
                "correct",
                'staining_contributions',
                'features'
            ],
        
        )
        self.outputs = pd.concat([self.outputs, outputs], ignore_index=True)

        if self.config.visualize:
            try:
                staining_contrib_dict={}

                for filename,staining_contribution in zip(filenames,staining_contributions):
                    staining_contrib_dict[filename[0].split("_")[-1].replace(".h5","")]=staining_contribution

                staining_contrib_dict=normalize_dict_values(staining_contrib_dict)
                patchwise_prediction=self.model.predict_single_patches(x)
                
                class_attention,patchwise_class_prob_of_prediction,attentions_normalized=self.model.get_class_attention(attentions,patchwise_prediction,preds)
                save_path_class_attention=self.attention_visualization(class_attention,batch,self.config,'class_attention_',plt.cm.viridis)
                # self.attention_visualization(patchwise_prediction,batch,self.config,"multiclass_",class_visualization=False)
                self.attention_visualization(patchwise_class_prob_of_prediction,batch,self.config,"class_",plt.cm.viridis)
                self.attention_visualization(attentions_normalized,batch,self.config,"attention_",plt.cm.viridis)
                save_path_report = Path(self.config.save_path) / self.config.name/ "reports"
                self.create_report_card(staining_contrib_dict,save_path_class_attention,patient,probs,save_path_report,y)
            except Exception as e:
                print(f"Could not create report for patient {patient}: {repr(e)}")



