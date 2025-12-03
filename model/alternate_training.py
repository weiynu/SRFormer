import argparse
import numpy as np
import pandas as pd
import os
from os.path import dirname
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
import copy

# root=dirname(os.getcwd())+"/lib/" #pycharm环境
root=os.getcwd()          #vscode设置
print(root)
sys.path.append(root)

sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from model.STAEformer import STAEformer


# ! X shape: (B, T, N, C)

DEVICE=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Refinement(nn.Module):
    def __init__(self, rate1=0.1, rate2=1.0, regularization=None):
        super(Refinement, self).__init__()

        # 改进3: 可学习的rate参数
        self.rate1 = nn.Parameter(torch.tensor(rate1))
        self.rate2 = nn.Parameter(torch.tensor(rate2))

    def forward(self, input, network_1, network_2,):
        B, T, N, D = input.shape
        # 改进4: 添加批量归一化
        bn = nn.BatchNorm2d(N).to(DEVICE)
        
        input_std = torch.std(input.detach()).to(DEVICE)
        scale = nn.Parameter(torch.tensor(float(input_std * 5.0)), requires_grad=False).to(DEVICE)
        ratio = nn.Parameter(torch.randn(B, T, N, 1) * 0.001).to(DEVICE)
        
        # 改进5: 使用更稳定的激活函数
        ratios = torch.sigmoid(ratio) * 0.1  # 限制噪声比例

        x = input.clone().to(DEVICE)
        # 改进6: 更智能的噪声生成 - 基于输入特征的重要性
        with torch.no_grad():
            feature_importance = torch.std(x, dim=(0, 1, 2), keepdim=True)
            normalized_importance = feature_importance / (torch.sum(feature_importance) + 1e-8)
        noise = torch.randn_like(x) * scale * normalized_importance
        # noise = torch.randn_like(x) * self.scale
        x_noise = x + ratios * noise

        x = x.transpose(1, 2)
        x = bn(x)
        x = x.reshape(B, N, T * D)

        x_noise = x_noise.transpose(1, 2)
        x_noise = bn(x_noise)
        x_noise = x_noise.reshape(B, N, T * D)

        y = network_1(x)
        y_noise = network_1(x_noise)

        yl = network_2(x)
        yl_noise = network_2(x_noise)

        # 改进8: 更稳定的损失计算
        loss = torch.mean((y_noise - y) ** 2) + 1e-8
        loss_l = torch.mean((yl_noise - yl) ** 2) + 1e-8

        # 改进9: 改进的熵计算，避免log(0)
        entropy = torch.mean(ratios * torch.log(ratios + 1e-8) +
                             (1 - ratios) * torch.log(1 - ratios + 1e-8))

        total_loss = loss + loss_l * self.rate2 - entropy * self.rate1
        # total_loss = loss  - entropy * self.rate1
        return total_loss, ratio

    def get_attention(self,ratio):
        # 原来的写法
        # ratios = torch.sigmoid(self.ratio)
        # return (1 - ratios / torch.max(ratios, dim=-1, keepdim=True)[0].reshape(self.ratio.shape)).softmax(dim=-1)

        # 修改
        # 改进11: 更稳定的注意力计算
        ratios = torch.sigmoid(ratio) * 0.1
        max_ratios = torch.max(ratios, dim=-1, keepdim=True)[0]
        normalized_ratios = ratios / (max_ratios + 1e-8)
        attention_weights = (1 - normalized_ratios)

        # 确保注意力权重和为1
        attention_sum = torch.sum(attention_weights, dim=-1, keepdim=True)
        return attention_weights / (attention_sum + 1e-8)


# -----------------------
# KL divergence KL(p || q) where p = attention_optimal, q = attention_model
# compute the similarity between attention map and their supervision per-batch mean
# -----------------------
def kl_divergence(p, q, eps=1e-8):
    # p, q: (B,T,N, 1), both prob distributions (sum to 1)
    # p = p.clamp(min=eps)
    # q = q.clamp(min=eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=1).mean()

@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch, att_s, embeddings = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch, att_s, embedding = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch, att_s, hidden = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss, att_s

def attention_refinement(model, refiner, dataloader, network_1, network_2, refiner_optimizer, refiner_scheduler, optimizer, scheduler):
    
    batch_loss_list=[]
    for x,y in dataloader:
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        with torch.no_grad():
            _, _, embeddings = model(x)
        
        refiner.to(DEVICE)
        refiner.train()
        min_refiner_loss=np.inf
        for r in range(30):
            refiner_optimizer.zero_grad()
            refine_loss, ratio = refiner(embeddings, network_1, network_2)
            refine_loss.backward()
            refiner_optimizer.step()
            if refine_loss < min_refiner_loss:
                min_refiner_loss = refine_loss
                optim_refiner_dict = copy.deepcopy(refiner.state_dict())
            refiner.load_state_dict(optim_refiner_dict)
        refiner_scheduler.step()
        optimal_att = refiner.get_attention(ratio)

        for name, param in model.named_parameters():
            if 'attn_layers_s' in name:
                param.requires_grad = True
        # for name, param in model.named_parameters():
        #     param.requires_grad = True
        model.train()
        out, att_model, embeddings = model(x)
        att_revised = torch.mean(att_model, dim=-1).reshape(optimal_att.shape)

        out = SCALER.inverse_transform(out)
        criterion = nn.HuberLoss()
        clf_loss = criterion(out, y)
        kl = kl_divergence(optimal_att, att_revised)
        gamma = 1.0  # weight of attention supervision
        # gamma = 0.5  # weight of attention supervision
        loss = clf_loss + gamma * kl
        batch_loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = np.mean(batch_loss_list)
    scheduler.step()
    return train_loss
   


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    network_1,
    network_2, 
    refiner_optimizer, 
    refiner_scheduler,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)
    # refiner=refiner.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        if (epoch+1)%7!=0:
            train_loss, train_att_s = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
            )
        else:
            train_loss=attention_refinement(model, refiner, trainset_loader, network_1, network_2, refiner_optimizer, refiner_scheduler, optimizer, scheduler)
            for p in model.parameters():
                p.requires_grad = True
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()

    seed = torch.randint(1000, (1,)) # set random seed here
    seed_everything(seed)
    # set_cpu_num(1)

    # GPU_ID = args.gpu_num
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"./data/{dataset}"
    print(data_path)
    model_name = STAEformer.__name__

    with open(os.getcwd()+f"/model/{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    model = STAEformer(**cfg["model_args"])
    refiner=Refinement().to(DEVICE)
    model_dim=cfg['model_args']['input_embedding_dim']+cfg['model_args']['tod_embedding_dim']+cfg['model_args']['dow_embedding_dim']+cfg['model_args']['spatial_embedding_dim']+cfg['model_args']['adaptive_embedding_dim']
    network_1=nn.Linear(cfg['model_args']['in_steps'] * model_dim, cfg['model_args']['out_steps'] * cfg['model_args']['output_dim']).to(DEVICE)
    network_2=nn.Linear(cfg['model_args']['in_steps'] * model_dim, cfg['model_args']['out_steps'] * cfg['model_args']['output_dim']).to(DEVICE)

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"./logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"./saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        # verbose=False,
    )

    refiner_optimizer = torch.optim.Adam(
        refiner.parameters(),
        lr=0.001,  # 学习率是不是大了，可试试0.001
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    refiner_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        refiner_optimizer,
        T_max=10,
        eta_min=1e-5,
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["batch_size"],
                cfg["in_steps"],
                cfg["num_nodes"],
                next(iter(trainset_loader))[0].shape[-1],
            ],
            verbose=0,  # avoid print twice
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        network_1,
        network_2, 
        refiner_optimizer, 
        refiner_scheduler,
        clip_grad=0,
        max_epochs=200,
        early_stop=10,
        verbose=1,
        plot=False,
        log=None,
        save=None,
    )

    print_log(f"Saved Model: {save}", log=log)

    test_model(model, testset_loader, log=log)

    log.close()
