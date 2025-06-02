import wandb

from .logger import Logger


class WandbLogger(Logger):
    def __init__(self, run_name: str, config: dict):
        wandb.init(
            entity="federated-flower-wanb",
            project="AD-Project",
            name=run_name,
            reinit=True,
            resume="allow",
            config=config,
        )

        wandb.define_metric("centralized/test_loss", step_metric="centralized/fed-round")
        wandb.define_metric("centralized/test_accuracy", step_metric="centralized/fed-round")

    @classmethod
    def log(cls, _: str, data: dict):
        wandb.log(data)
