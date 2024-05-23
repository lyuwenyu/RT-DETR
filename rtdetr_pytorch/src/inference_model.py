from torch import nn


class InferenceModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        print(self.postprocessor.deploy_mode)

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)
