from torch.optim import lr_scheduler

class SchedulerManager:
    @staticmethod
    def build_scheduler(optimizer, cfg):
        """
        Return a learning rate scheduler
            For 'linear', we keep the same learning rate for the first <cfg.niter_warmup> epochs
        and linearly decay the rate to zero over the next <cfg.niter_decay> epochs.
        For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
        See https://pytorch.org/docs/stable/optim.html for more details.
        Args:
            optimizer (torch.optim): the optimizer of the network
            cfg: cfg.lr_decay_method is the name of the decay method of learning rate: linear | step | plateau | cosine
        return: scheduler
        """
        if cfg.lr_decay_method == "step":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_niters, gamma=cfg.gama)
        elif cfg.lr_decay_method == "plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
        elif cfg.lr_decay_method == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_iter, eta_min=0)
        else:
            raise NotImplementedError("lr_decay_method '{}' is not implemented".format(cfg.lr_decay_method))
