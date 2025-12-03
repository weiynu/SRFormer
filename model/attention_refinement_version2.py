import copy

import torch.optim as optim
import numpy as np
import os
from os.path import dirname
import torch
import torch.nn as nn
import yaml
import sys
from torch.optim import Adam


root=dirname(os.getcwd())+"/lib/" #pycharm环境
sys.path.append(root)
sys.path.append("..")

from lib.data_prepare import get_dataloaders_from_index_data
from train import RMSE_MAE_MAPE
from STAEformer import STAEformer as STAE

class Refinement(nn.Module):
    def __init__(self, input, input_label,network_1, network_2,
                 rate1=0.1, rate2=1.0, regularization=None):
        super(Refinement, self).__init__()

        self.input_sentences = input
        self.label = input_label
        self.output_dim=1

        self.batch_size, self.in_steps, self.num_nods, self.dimension = input.shape
        self.out_steps = 12

        self.ratio = nn.Parameter(torch.randn(self.batch_size, self.in_steps, self.num_nods, 1) * 0.01)

        self.scale = torch.tensor(float(np.std(input.detach().cpu().numpy()) * 10.0))

        self.network_1 = network_1
        self.network_2 = network_2

        self.rate1 = rate1
        self.rate2 = rate2

    def forward(self,x):
        B, T, N, D= x.shape
        ratios = torch.sigmoid(self.ratio)

        x = self.input_sentences.clone()
        noise = torch.randn_like(x) * self.scale
        x_noise = x + ratios * noise

        x = x.transpose(1, 2)
        x = x.reshape(B, N, T * D)

        x_noise = x_noise.transpose(1, 2)
        x_noise = x_noise.reshape(B, N, T * D)

        y = self.network_1(x)
        y_noise = self.network_1(x_noise)

        yl = self.network_2(x)
        yl_noise = self.network_2(x_noise)

        loss = torch.mean((y_noise - y) ** 2)
        loss_l = torch.mean((yl_noise - yl) ** 2)

        entropy = torch.mean(torch.log(ratios + 1e-8))

        total_loss = loss + loss_l * self.rate2 - entropy * self.rate1
        return total_loss

    def get_attention(self):
        ratios = torch.sigmoid(self.ratio)
        return (1 - ratios / torch.max(ratios, dim=-1, keepdim=True)[0].reshape(self.ratio.shape)).softmax(dim=-1)

# -----------------------
# load pretrained STAEformer as the target of refinement
# -----------------------
def load_model(model_path, device):
    model_name = STAE.__name__
    with open(f"./{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg['PEMS08']
    model = STAE(**cfg["model_args"])
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)

# -----------------------
# KL divergence KL(p || q) where p = attention_optimal, q = attention_model
# compute the similarity between attention map and their supervision per-batch mean
# -----------------------
def kl_divergence(p, q, eps=1e-8):
    # p, q: (B,T,N, 1), both prob distributions (sum to 1)
    # nn.KLDivLoss(reduction='batchmean')
    # p = p.clamp(min=eps)
    # q = q.clamp(min=eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=1).mean()

@torch.no_grad()
def predict(model, loader,device):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        out_batch, att_s, hidden = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out

@torch.no_grad()
def eval_model(model, valset_loader, criterion,device):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        out_batch, att_s, embeddings = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    data_path = f"../data/PEMS08"
    model_name = STAE.__name__
    model_path = f"../saved_models/STAEformer-PEMS08-2025-10-27-21-58-25.pt"
    in_steps = 12
    model_dim = 152
    out_steps = 12
    output_dim = 1

    network_1 = nn.Linear(in_steps * model_dim, out_steps * output_dim).to(device)
    network_2 = nn.Linear(in_steps * model_dim, out_steps * output_dim).to(device)

    with open(f"./{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg['PEMS08']

    trainset_loader, valset_loader, testset_loader, SCALER, = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
    )

    refine_model=load_model(model_path, device)
    for p in refine_model.parameters():
        p.requires_grad=True
    refine_model.train()

    refine_model_optimizer = torch.optim.Adam(
        refine_model.parameters(),
        lr=cfg['lr']
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        refine_model_optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        # verbose=False,
    )

    min_val_loss=np.inf
    wait=0
    train_loss_list = []
    val_loss_list = []

    for epoch in range(cfg["max_epochs"]):
        batch_loss_list=[]
        for x, y in trainset_loader:
            x=x.to(device)
            y=y.to(device)
            with torch.no_grad():
                out, att_model, embeddings = refine_model(x)

            refiner=Refinement(embeddings, y, network_1, network_2).to(device)
            refiner_optimizer = optim.Adam(
                refiner.parameters(),
                lr=0.01,
                weight_decay=cfg.get("weight_decay", 0),
                eps=cfg.get("eps", 1e-8),
            )
            refiner_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                refiner_optimizer,
                milestones=cfg["milestones"],
                gamma=cfg.get("lr_decay_rate", 0.1),
                # verbose=False,
            )
            refiner.train()

            for r in range(30):
                refiner_optimizer.zero_grad()
                refine_loss=refiner(embeddings)
                refine_loss.backward()
                refiner_optimizer.step()
                refiner_scheduler.step()

            optimal_att=refiner.get_attention()
            out, att_model, embeddings = refine_model(x)
            att_revised=torch.mean(att_model, dim=-1).reshape(optimal_att.shape)


            out=SCALER.inverse_transform(out)
            criterion = nn.HuberLoss()
            clf_loss = criterion(out, y)
            kl = kl_divergence(optimal_att, att_revised)
            # gamma = 1.0  # weight of attention supervision
            gamma = 0.5  # weight of attention supervision
            loss = clf_loss + gamma * kl
            batch_loss_list.append(loss.item())
            refine_model_optimizer.zero_grad()
            loss.backward()
            refine_model_optimizer.step()

        train_loss=np.mean(batch_loss_list)
        scheduler.step()

        train_loss_list.append(train_loss)
        val_loss=eval_model(refine_model, valset_loader, criterion, device)
        val_loss_list.append(val_loss)

        if val_loss < min_val_loss:
            min_val_loss=val_loss
            best_epoch=epoch
            best_state_dict=copy.deepcopy(refine_model.state_dict())
        else:
            wait+=1
            if wait>cfg["early_stop"]:
                break
        refine_model.load_state_dict(best_state_dict)

        test_rmse, test_mae, test_mape = RMSE_MAE_MAPE(*predict(refine_model, testset_loader, device))
        print(f'EPOCH={epoch}, LOSS={train_loss:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.4f}')

