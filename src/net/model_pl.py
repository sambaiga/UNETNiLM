import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import sys
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from .modules import CNN1DModel,  UNETNiLM
from net.metrics import  compute_metrics, example_f1_score, compute_regress_metrics
from data.load_data import ukdale_appliance_data, refit_appliance_data
from data.data_loader import Dataset, load_data, spilit_refit_test
from .utils import ObjectDict, QuantileLoss
from pytorch_lightning.metrics.functional import f1_score



class NILMnet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = ObjectDict()
        self.hparams.update(hparams.__dict__ if hasattr(hparams, '__dict__') else hparams)
        self._data = None
        self.q_criterion = QuantileLoss(self.hparams.quantiles)
        if self.hparams.model_name== "CNN1D":
            self.model = CNN1DModel(in_size=self.hparams.in_size, 
                               num_classes=self.hparams.out_size,
                               d_model=self.hparams.d_model,
                                dropout=self.hparams.dropout, 
                               seq_size=self.hparams.seq_len,  
                               n_layers=self.hparams.n_layers, 
                               n_quantiles=len(self.hparams.quantiles),
                               mc=self.hparams.mc,
                               pos_enconding=self.hparams.pos_enconding)
        
        elif self.hparams.model_name=="UNETNiLM":
            self.model =  UNETNiLM(n_channels=self.hparams.in_size, 
                               num_classes=self.hparams.out_size,
                               features_start=self.hparams.d_model//4,
                               seq_size=self.hparams.seq_len, 
                               num_layers=self.hparams.n_layers,
                               n_quantiles=len(self.hparams.quantiles),
                               mc=self.hparams.mc,
                               pos_enconding=self.hparams.pos_enconding,
                               dropout=self.hparams.dropout
                               )  
            
    def forward(self, x):
            return self.model(x)        
        
    def _step(self, batch):
        x, y, z = batch
        B, T = y.size()
        logits, rmse_logits = self(x)
        loss_nll   = F.nll_loss(F.log_softmax(logits, 1), z)
        if len(self.hparams.quantiles)>1:
            loss_mse = self.q_criterion(rmse_logits, y)
            mae_score = F.l1_loss(rmse_logits,y.unsqueeze(1).expand_as(rmse_logits))
        else:    
            loss_mse = F.mse_loss(rmse_logits, y)
            mae_score = F.l1_loss(rmse_logits, y)
            
        loss = loss_nll + loss_mse
        pred = torch.max(F.softmax(logits, 1), 1)[1]
        res = f1_score(pred, z)
        logs = {"nlloss":loss_nll, "mseloss":loss_mse,
                 "mae":mae_score, "F1": res}
        return loss, logs
    
    def training_step(self, batch, batch_idx):
        loss , logs = self._step(batch)
        train_logs = {}
        for key, value in logs.items():
            train_logs[f'tra_{key}']=value.item()
        return {'loss': loss, 'log': train_logs}
    
    def validation_step(self, batch, batch_idx):
        loss , logs = self._step(batch)
        return logs
    
    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = np.mean([x['nlloss'].item()+x['mseloss'].item() for x in outputs])
        avg_f1 = np.mean([x['F1'].item() for x in outputs])
        avg_rmse = np.mean([x['mae'].item() for x in outputs])
        logs = {'val_loss': avg_loss, "val_F1": avg_f1, "val_mae":avg_rmse}
        return {'log':logs}
    
    def _mcdropout_step(self, batch):
        x, y, z = batch
        B, T = y.size()
        logit_state_samples = torch.zeros(self.hparams.n_model_samples, B, 2, T).to(y.device)
        power_samples = torch.zeros(self.hparams.n_model_samples, B, T).to(y.device)
        # sampling the model, then z and classifying
        for k in range(self.hparams.n_model_samples):
            logits, pred_power  = self(x)
            logits =  F.softmax(logits, 1)
            logit_state_samples[k] = logits
            power_samples[k] = pred_power
            
        logit_state_mean = torch.mean(logit_state_samples, dim=0) 
        power_mean = torch.mean(power_samples, dim=0)
        logit_state_std = torch.std(logit_state_samples, dim=0)
        power_std       = torch.std(power_samples, dim=0)
        
        logs = {"sample_power":power_samples, 
                "sample_state":logit_state_samples,
                "mu_power":power_mean, 
                "mu_state":logit_state_mean,
                "std_power":power_std, 
                "std_state":power_std,
                 "power":y, "state":z}
        return logs
    
    
    def test_step(self, batch, batch_idx):
        x, y, z = batch
        if  self.hparams.mc:
            logs = self._mcdropout_step(batch)
        else:
            B, T = y.size()
            logits, pred_power  = self(x)
            logits_state = F.softmax(logits, 1)
            logs = {"pred_power":pred_power, "logits_state":logits_state, "power":y, "state":z}
        return logs
    
    def test_epoch_end(self, outputs):
        
        power = torch.cat([x['power'] for x in outputs], 0).cpu().numpy()
        state = torch.cat([x['state'] for x in outputs], 0).cpu().numpy().astype(np.int32)
    
        if  self.hparams.mc:
            #mc dropout and batch uncertanity according to 
            #1. https://github.com/icml-mcbn/mcbn: Bayesian Uncertainty Estimation for Batch Normalized Deep Networks,
            #2. https://arxiv.org/pdf/1506.02142.pdf: Dropout as a Bayesian Approximation:Representing Model Uncertainty in Deep Learning
            
            #pred_power_samples = torch.cat([x['sample_power'] for x in outputs], 0).data.cpu().numpy()
            #pred_state_samples = torch.cat([x['sample_state'] for x in outputs], 0).data.cpu().numpy()
            
            pred_mu_power  = torch.cat([x['mu_power'] for x in outputs], 0).data.cpu().numpy()
            pred_std_power = torch.cat([x['std_power'] for x in outputs], 0).data.cpu().numpy()
            
            pred_logits_mean = torch.cat([x['mu_state'] for x in outputs], 0).data
            pred_logits_std = torch.cat([x['std_state'] for x in outputs], 0).data
            pred_mu_prob, pred_mu_states = torch.max(pred_logits_mean, 1)
            pred_std_prob, pred_std_states = torch.max(pred_logits_std, 1)
            
            pred_mu_states = pred_mu_states.cpu().numpy().astype(np.int32)
            pred_std_states = pred_std_states.cpu().numpy()
            
            pred_mu_prob = pred_mu_prob.cpu().numpy()
            pred_std_prob = pred_std_prob.cpu().numpy()
            
            
            class_results, regress_results=self.get_results(pred_mu_power,  pred_mu_states, power, state)
            
            logs = {"pred_mu_power":pred_mu_power, 
                    "pred_std_power":pred_std_power,
                     "pred_mu_state":pred_mu_states,
                     "pred_std_state":pred_std_states, 
                     "prob_mu_state":pred_mu_prob, 
                     "prob_std_state":pred_std_prob, 
                     "power":power, "state":state,  
                     'mlabel_results':class_results, 
                     'reg_results':regress_results} 
        
            
        else:    
            pred_power = torch.cat([x['pred_power'] for x in outputs], 0).cpu().numpy()
            pred_logits = torch.cat([x['logits_state'] for x in outputs], 0)
            prob_states, pred_states = torch.max(pred_logits, 1)
            prob_states = prob_states.cpu().numpy()
            pred_states = pred_states.cpu().numpy().astype(np.int32)
            if len(self.hparams.quantiles)>2:
                idx = len(self.hparams.quantiles)//2
                class_results, regress_results=self.get_results(pred_power[:,idx],  
                                                                pred_states, 
                                                                power, 
                                                                state)
            else:
                class_results, regress_results=self.get_results(pred_power,  
                                                                pred_states, 
                                                                power, 
                                                                state)    
            logs = {"pred_power":pred_power, 
                     "pred_state":pred_states, 
                      "prob_state":prob_states, 
                        "power":power, "state":state,  
                         'mlabel_results':class_results, 
                            'reg_results':regress_results} 
        
        
        
        
       
        return logs
     
    def get_results(self, pred_power, pred_state, power, state):    
        import pandas as pd
        appliance_data = ukdale_appliance_data if self.hparams.data=="ukdale" else refit_appliance_data
        regress_results = []
        for idx, app in enumerate(list(appliance_data.keys())):
            pred_power[:,idx] = (pred_power[:,idx] * appliance_data[app]['std']) + appliance_data[app]['std']
            pred_power[:,idx] = np.where(pred_power[:,idx]<0, 0, pred_power[:,idx])
            power[:,idx] = (power[:,idx] * appliance_data[app]['std']) + appliance_data[app]['std']
            result = compute_regress_metrics(power[:,idx], pred_power[:,idx])
            result = pd.DataFrame.from_dict(result, orient="index")
            regress_results.append(result)
        regress_results = pd.concat(regress_results, axis=1)
        regress_results.columns = list(appliance_data.keys())
        classification_results = compute_metrics(state, pred_state)
        classification_results=pd.DataFrame.from_dict(classification_results, orient="index")
        return classification_results, regress_results
        
    def configure_optimizers(self):  
        optim = torch.optim.Adam(self.parameters(),lr=self.hparams.learning_rate, betas=(self.hparams.beta_1, self.hparams.beta_2))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=self.hparams.patience_scheduler, verbose=True, min_lr=1e-6, mode="max") # note early stopping has patient 3
        scheduler = {'scheduler':sched, 
                     'monitor': 'val_F1',
                     'interval': 'epoch',
                     'frequency': 1}
        return [optim], [scheduler]
        
    
    @pl.data_loader
    def train_dataloader(self):
        
        data = Dataset(self._get_cache_data()['x_train'], self._get_cache_data()['y_train'], 
                             self._get_cache_data()['z_train'],  seq_len=self.hparams.seq_len)
        return torch.utils.data.DataLoader(data,batch_size=self.hparams.batch_size,
                                            shuffle=True,pin_memory=True,
                                            num_workers=self.hparams.num_workers)
        
    @pl.data_loader
    def val_dataloader(self):
        
        data = Dataset(self._get_cache_data()['x_val'], self._get_cache_data()['y_val'], 
                             self._get_cache_data()['z_val'],  seq_len=self.hparams.seq_len)
        return torch.utils.data.DataLoader(data,batch_size=self.hparams.batch_size,
                                            shuffle=False,pin_memory=True,
                                            num_workers=self.hparams.num_workers)  
    @pl.data_loader     
    def test_dataloader(self):
        
        data = Dataset(self._get_cache_data()['x_test'], self._get_cache_data()['y_test'], 
                             self._get_cache_data()['z_test'],  seq_len=self.hparams.seq_len)
        return torch.utils.data.DataLoader(data,batch_size=self.hparams.batch_size,
                                            shuffle=False,pin_memory=True,
                                            num_workers=self.hparams.num_workers)        
    
    def _get_cache_data(self):
        if self._data is None:
            '''
            if self.hparams.data =="ukdale":
                x_train, y_train, z_train = load_data(data_path=self.hparams.data_path, 
                                                    data_type="training", 
                                                    sample=self.hparams.sample,
                                                    data=self.hparams.data,
                                                    denoise=self.hparams.denoise)
                
                x_test, y_test, z_test = load_data(data_path=self.hparams.data_path, 
                                                    data_type="test", 
                                                    sample=self.hparams.sample,
                                                     data=self.hparams.data,
                                                    denoise=self.hparams.denoise)
                x_val, y_val, z_val = load_data(data_path=self.hparams.data_path, 
                                                    data_type="validation", 
                                                    sample=self.hparams.sample,
                                                     data=self.hparams.data,
                                                    denoise=self.hparams.denoise)
            '''                                        
           
            x , y , z = load_data(data_path=self.hparams.data_path, 
                                                    data_type="test" if self.hparams.data=="refit" else "training", 
                                                    sample=self.hparams.sample,
                                                     data=self.hparams.data,
                                                    denoise=self.hparams.denoise) 
            x_train, x_val, x_test = spilit_refit_test(x)
            y_train, y_val, y_test = spilit_refit_test(y)
            z_train, z_val, z_test = spilit_refit_test(z)       
           
           
           
            self._data = dict(x_test=x_test, y_test=y_test, z_test=z_test,
                              x_val=x_val, y_val=y_val, z_val=z_val,
                              x_train=x_train, y_train=y_train, z_train=z_train)
            
        return self._data   
    
    
    @staticmethod
    def add_model_specific_args():
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(add_help=False)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--beta_1', default=0.999, type=float)
        parser.add_argument('--beta_2', default= 0.98, type=float)
        parser.add_argument('--eps', default=1e-8, type=float)
        parser.add_argument('--patience_scheduler', default=5, type=int)
        parser.add_argument('--dropout', default=0.25, type=float)
        parser.add_argument('--d_model', default=128, type=int)
        parser.add_argument('--pool_filter', default=8, type=int)
        parser.add_argument('--n_layers', default=5, type=int)
        parser.add_argument('--seq_len', default=99, type=int)
        parser.add_argument('--out_size', default=5, type=int)
        parser.add_argument('--in_size', default=1, type=int)
        parser.add_argument('--denoise', default=False, type=bool)
        parser.add_argument('--mc', default=False, type=bool)
        parser.add_argument('--pos_enconding', default=True, type=bool)
        parser.add_argument('--num_head', default=8, type=int)
        parser.add_argument('--model_name', default="CNN1D", type=str)
        parser.add_argument('--data', default="ukdale", type=str)
        parser.add_argument('--quantiles', default=[0.0025,0.1, 0.5, 0.9, 0.975], type=list)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--n_model_samples', default=100, type=int)
        
        #parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
        return parser
