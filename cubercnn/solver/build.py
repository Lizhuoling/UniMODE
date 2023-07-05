import pdb
import torch
import math
from typing import Any, Dict, List, Set
from detectron2.solver.build import maybe_add_gradient_clipping
from torch.nn.parallel import DistributedDataParallel
import torch.optim.lr_scheduler as lr_sched

from mmcv.runner.optimizer import DefaultOptimizerConstructor

def build_optimizer(cfg, model):
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    # For RCNN3D
    if cfg.SOLVER.TYPE == 'mmcv_AdamW':
        optimizer = dict(type='AdamW', lr=cfg.SOLVER.BASE_LR, weight_decay=0.01)
        paramwise_cfg = dict(custom_keys={'img_backbone': dict(lr_mult=0.1)})
        optimizer_constructor = DefaultOptimizerConstructor(optimizer, paramwise_cfg)
        optimizer = optimizer_constructor(model)

    elif cfg.SOLVER.TYPE == 'sgd':
        for module in model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY

                if isinstance(module, norm_module_types) and (cfg.SOLVER.WEIGHT_DECAY_NORM is not None):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
                
                elif key == "bias":
                    if (cfg.SOLVER.BIAS_LR_FACTOR is not None):
                        lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                    if (cfg.SOLVER.WEIGHT_DECAY_BIAS is not None):
                        weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

                # these params do not need weight decay at all
                # TODO parameterize these in configs instead.
                if key in ['priors_dims_per_cat', 'priors_z_scales', 'priors_z_stats']:
                    weight_decay = 0.0

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # For PETR
    elif cfg.SOLVER.TYPE == 'adamw':
        lr = cfg.SOLVER.BASE_LR
        lr_mult = 0.25
        if type(model) == DistributedDataParallel: model = model.module
        backbone_params_ids = list(map(id, model.detector.img_backbone.parameters()))
        normal_params = filter(lambda p: id(p) not in backbone_params_ids, model.parameters())
        params = [
            {"params": normal_params, "lr": lr},
            {"params": model.detector.img_backbone.parameters(), "lr": lr_mult * lr},
        ]

    elif cfg.SOLVER.TYPE == 'sgd':
        optimizer = torch.optim.SGD(
            params, 
            cfg.SOLVER.BASE_LR, 
            momentum=cfg.SOLVER.MOMENTUM, 
            nesterov=cfg.SOLVER.NESTEROV, 
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.TYPE == 'adam':
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR, eps=1e-02)
    elif cfg.SOLVER.TYPE == 'adam+amsgrad':
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR, amsgrad=True, eps=1e-02)
    elif cfg.SOLVER.TYPE == 'adamw':
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR, eps=1e-02)
    elif cfg.SOLVER.TYPE == 'adamw+amsgrad':
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR, amsgrad=True, eps=1e-02)
    else:
        raise ValueError('{} is not supported as an optimizer.'.format(cfg.SOLVER.TYPE))

    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer

def freeze_bn(network):

    for _, module in network.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = False

def build_lr_scheduler(cfg, optimizer):
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME

    if name == "CosineAnnealing":
        total_epochs = math.ceil(cfg.SOLVER.MAX_ITER / cfg.SOLVER.VIRTUAL_EPOCH_PER_ITERATION)
        lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=total_epochs)

    elif name == "WarmupMultiStepLR":
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                "These values will be ignored."
            )
        sched = MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA**k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    elif name == "WarmupCosineLR":
        end_value = cfg.SOLVER.BASE_LR_END / cfg.SOLVER.BASE_LR
        assert end_value >= 0.0 and end_value <= 1.0, end_value
        sched = CosineParamScheduler(1, end_value)
    elif name == "WarmupStepWithFixedGammaLR":
        sched = StepWithFixedGammaParamScheduler(
            base_value=1.0,
            gamma=cfg.SOLVER.GAMMA,
            num_decays=cfg.SOLVER.NUM_DECAYS,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    if name == "CosineAnnealing":
        lr_warmup_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.SOLVER.WARMUP_ITERS, warmup_ratios=cfg.SOLVER.WARMUP_FACTOR)
        return lr_scheduler, lr_warmup_scheduler
    else:
        sched = WarmupParamScheduler(
            sched,
            cfg.SOLVER.WARMUP_FACTOR,
            min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
            cfg.SOLVER.WARMUP_METHOD,
            cfg.SOLVER.RESCALE_INTERVAL,
        )
        return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)

class LinearWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, warmup_steps, warmup_ratios=0.3333, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.warmup_ratios = warmup_ratios
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        k = (1 - self.last_epoch / self.warmup_steps) * (1 - self.warmup_ratios)

        return [base_lr * (1-k) for base_lr in self.base_lrs]
