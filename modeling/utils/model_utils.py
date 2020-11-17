import torch


class ModelManager:
    """
    Initialize a model:
        register CPU/ GPU device (with multi-GPU support)
    """
    @staticmethod
    def init_model(model, gpu_ids=[]):
        """
        Args:
            model (torch.nn.Module): the model which implements torch.nn.Module
            gpu_ids (list[int]): the training process running on which GPU
        return:
            model
        """
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            model.to(gpu_ids[0])
            model = torch.nn.DataParallel(model, gpu_ids)
        return model


