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
                               output_size=self.hparams.out_size,
                               d_model=self.hparams.d_model,
                                dropout=self.hparams.dropout, 
                               seq_len=self.hparams.seq_len,  
                               n_layers=self.hparams.n_layers, 
                               n_quantiles=len(self.hparams.quantiles),
                               pool_filter=self.hparams.pool_filter)
        
        elif self.hparams.model_name=="UNETNiLM":
            self.model =  UNETNiLM(in_size=self.hparams.in_size, 
                               output_size=self.hparams.out_size,
                               features_start=self.hparams.d_model//4,
                               seq_len=self.hparams.seq_len, 
                               n_layers=self.hparams.n_layers,
                               n_quantiles=len(self.hparams.quantiles),
                               pool_filter=self.hparams.d_model//4
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
    
    def test_step(self, batch, batch_idx):
        x, y, z = batch
        B, T = y.size()
        logits, pred_power  = self(x)
        pred_state = torch.max(F.softmax(logits, 1), 1)[1]
        logs = {"pred_power":pred_power, "pred_state":pred_state, "power":y, "state":z}
        return logs
    
    def test_epoch_end(self, outputs):
        
        pred_power = torch.cat([x['pred_power'] for x in outputs], 0).cpu().numpy()
        pred_state = torch.cat([x['pred_state'] for x in outputs], 0).cpu().numpy().astype(np.int32)
        power = torch.cat([x['power'] for x in outputs], 0).cpu().numpy()
        state = torch.cat([x['state'] for x in outputs], 0).cpu().numpy().astype(np.int32)
        
        appliance_data = ukdale_appliance_data if self.hparams.data=="ukdale" else refit_appliance_data
        import pandas as pd
        print(pred_power.shape)
        if len(self.hparams.quantiles)>1:
            regress_results = {}
            for idx, app in enumerate(list(appliance_data.keys())):
                quantile_regress_results = []
                power[:,idx] = (power[:, idx] * appliance_data[app]['std']) + appliance_data[app]['std']
                for q_i, q in enumerate(self.hparams.quantiles):
                    pred_power[:,q_i, idx] = (pred_power[:,q_i, idx] * appliance_data[app]['std']) + appliance_data[app]['std']
                    pred_power[:,q_i, idx] = np.where(pred_power[:,q_i, idx]<0, 0, pred_power[:,q_i, idx])
                    result = compute_regress_metrics(power[:,idx], pred_power[:,q_i, idx])
                    result = pd.DataFrame.from_dict(result, orient="index")
                    quantile_regress_results.append(result)
                quantile_regress_results = pd.concat(quantile_regress_results, axis=1) 
                quantile_regress_results.columns = [str(round(q*100,2)) for q in self.hparams.quantiles]   
                regress_results[app] = quantile_regress_results
                     
        else:    
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
            regress_results = regress_results.append(exbF1)  
        
        #exbF1={"exbF1":example_f1_score(state,pred_state, per_sample=True, axis=0)} 
        #exbF1=pd.DataFrame.from_dict(exbF1, orient="index")  
        #exbF1.columns = list(appliance_data.keys())
        results = compute_metrics(state, pred_state, all_metrics=True, verbose=False)
        results=pd.DataFrame.from_dict(results, orient="index")
        logs = {"pred_power":pred_power, "pred_state":pred_state, 
                "power":power, "state":state,  'results':results, 'reg_results':regress_results}
        return logs
        
        
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
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--beta_1', default=0.999, type=float)
        parser.add_argument('--beta_2', default= 0.98, type=float)
        parser.add_argument('--eps', default=1e-8, type=float)
        parser.add_argument('--patience_scheduler', default=5, type=int)
        parser.add_argument('--weight_decay', default=0.0005, type=float)
        parser.add_argument('--dropout', default=0.25, type=float)
        parser.add_argument('--d_model', default=128, type=int)
        parser.add_argument('--pool_filter', default=8, type=int)
        parser.add_argument('--n_layers', default=5, type=int)
        parser.add_argument('--seq_len', default=99, type=int)
        parser.add_argument('--out_size', default=5, type=int)
        parser.add_argument('--in_size', default=1, type=int)
        parser.add_argument('--denoise', default=False, type=bool)
        parser.add_argument('--num_head', default=8, type=int)
        parser.add_argument('--model_name', default="CNN1D", type=str)
        parser.add_argument('--data', default="ukdale", type=str)
        parser.add_argument('--quantiles', default=[0.0025,0.1, 0.5, 0.9, 0.975], type=list)
        parser.add_argument('--num_workers', default=4, type=int)
        #parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
        return parser
