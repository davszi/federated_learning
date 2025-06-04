import wandb

from .logger import Logger


class WandbLogger(Logger):
    def __init__(self, run_name: str, config: dict | None = None):
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
    def log(cls, data: dict, name: str = '', step: int | float = None):
        if name != '':
            data = {
                f"{name}/{key}": value for key, value in data.items()
            }
        wandb.log(data, step=step)
