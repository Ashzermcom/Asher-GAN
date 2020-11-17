from torch.nn import init


class WeightManager:
    """
    Initialize network weights
    cfg.init_method is the name of the initialization method. normal | xavier | kaiming | orthogonal
    cfg.scale_factor is the scaling factor of method normal, xavier and orthogonal.
    """
    @staticmethod
    def init_weight(model, cfg):
        """
        Args:
            model: pytorch network which implements form torch.nn.Module
            cfg: config file
        """
        def init_func(m):
            class_name = m.__class__.__name__
            if hasattr(m, "weight") and (class_name.find("Conv") != -1 or class_name.find("Linear") != -1):
                if cfg.init_method == "normal":
                    init.normal_(m.weight.data, 0.0, cfg.scale_factor)
                elif cfg.init_method == "xavier":
                    init.xavier_normal_(m.weight.data, gain=cfg.scale_factor)
                elif cfg.init_method == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif cfg.init_method == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=cfg.scale_factor)
                else:
                    raise NotImplementedError("initialization method {} is not implemented".format(cfg.init_method))

                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

                elif class_name.find('BatchNorm2d') != -1:
                    # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                    init.normal_(m.weight.data, 1.0, cfg.scale_factor)
                    init.constant_(m.bias.data, 0.0)
            print("# ---------- Weight Initialization ---------- #")
            print("initial weight with method: '{}'".format(cfg.init_method))
            print("# ---------- Initial  Successfully ---------- #")
            model.apply(init_func)



