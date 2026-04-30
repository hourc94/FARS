import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn import metrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None and self.alpha >= 0:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_factor * focal_factor * bce
        else:
            loss = focal_factor * bce

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            neg_probs = torch.clamp(neg_probs + self.clip, max=1.0)

        pos_loss = targets * torch.log(torch.clamp(pos_probs, min=self.eps))
        neg_loss = (1.0 - targets) * torch.log(torch.clamp(neg_probs, min=self.eps))

        pos_weight = (1.0 - pos_probs).pow(self.gamma_pos) * targets
        neg_weight = (1.0 - neg_probs).pow(self.gamma_neg) * (1.0 - targets)
        loss = -(pos_weight * pos_loss + neg_weight * neg_loss)

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


def _move_sim_data(sim_data, device):
    if sim_data is None:
        return None

    def move_value(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(move_value(item) for item in value)
        if isinstance(value, list):
            return [move_value(item) for item in value]
        return value

    for key, value in vars(sim_data).items():
        setattr(sim_data, key, move_value(value))
    return sim_data


def build_scheduler(optimizer, args):
    if args.scheduler == 'none':
        return None
    if args.scheduler == 'cosine':
        t_max = args.scheduler_t_max if args.scheduler_t_max > 0 else args.epochs
        return CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=args.scheduler_eta_min
        )
    raise ValueError('Unsupported scheduler: {}'.format(args.scheduler))


def build_loss_function(args):
    if args.loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    if args.loss_type == 'focal':
        return BinaryFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    if args.loss_type == 'asl':
        return AsymmetricLoss(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip
        )
    raise ValueError('Unsupported loss type: {}'.format(args.loss_type))


def train_epochs(train_dataset, test_dataset, model, args, sim_data=None):
    num_workers = args.num_workers
    pin_memory = (args.device.type == 'cuda') and (not args.disable_pin_memory)
    base_seed = int(getattr(args, 'current_seed', getattr(args, 'seed', 0)))
    fold_seed = int(getattr(args, 'current_fold', 0))
    loader_seed = base_seed * 100 + fold_seed
    loader_generator = torch.Generator()
    loader_generator.manual_seed(loader_seed)

    def seed_worker(worker_id):
        worker_seed = loader_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'worker_init_fn': seed_worker,
    }
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              generator=loader_generator, **loader_kwargs)

    test_loader = DataLoader(test_dataset, args.test_batch_size, shuffle=False,
                             **loader_kwargs)

    model.to(args.device).reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = build_scheduler(optimizer, args)
    sim_data = _move_sim_data(sim_data, args.device)

    start_epoch = 1
    pbar = range(start_epoch, args.epochs + start_epoch)
    best_epoch, best_auc, best_aupr = 0, 0, 0
    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, args.device, args, sim_data)
        if epoch % args.valid_interval == 0:
            roc_auc, aupr = evaluate_metric(model, test_loader, args.device, epoch, sim_data)
            current_lr = optimizer.param_groups[0]['lr']
            print("epoch {}".format(epoch), "train_loss {0:.4f}".format(train_loss),
                  "lr {0:.6g}".format(current_lr),
                  "roc_auc {0:.4f}".format(roc_auc), "aupr {0:.4f}".format(aupr))
            if roc_auc > best_auc:
                best_epoch, best_auc, best_aupr = epoch, roc_auc, aupr
        if scheduler is not None:
            scheduler.step()

    print("best_epoch {}".format(best_epoch), "best_auc {0:.4f}".format(best_auc),
          "aupr {0:.4f}".format(best_aupr))

    return best_auc, best_aupr


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    elif getattr(data, 'x', None) is not None:
        return data.x.size(0)
    elif getattr(data, 'y', None) is not None:
        return data.y.view(-1).size(0)
    else:
        raise ValueError('Unable to infer batch size from data object.')


def train(model, optimizer, loader, device, args, sim_data=None):
    model.train()
    total_loss = 0
    pbar = loader
    loss_function = build_loss_function(args)

    for data in pbar:
        optimizer.zero_grad()
        true_label = data.to(device)
        if sim_data is not None and hasattr(model, 'moe_fusion'):
            predict = model(true_label, sim_data)
        else:
            predict = model(true_label, sim_data) if sim_data is not None else model(true_label)

        loss = loss_function(predict, true_label.y.view(-1))

        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()

    return total_loss / len(loader.dataset)


def evaluate_metric(model, loader, device, epoch, sim_data=None):
    model.eval()
    all_y_true = []
    all_y_score = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data, sim_data) if sim_data is not None else model(data)

            y_true = data.y.view(-1).cpu().numpy()
            y_score = out.cpu().numpy()

            # Collect all prediction results and true labels
            all_y_true.append(y_true)
            all_y_score.append(y_score)

    all_y_true = np.concatenate(all_y_true)
    all_y_score = np.concatenate(all_y_score)

    fpr, tpr, _ = metrics.roc_curve(all_y_true, all_y_score)
    roc_auc = metrics.auc(fpr, tpr)

    precision_curve, recall_curve, _ = metrics.precision_recall_curve(all_y_true, all_y_score)
    aupr = metrics.auc(recall_curve, precision_curve)

    return roc_auc, aupr
