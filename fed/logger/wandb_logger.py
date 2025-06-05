import wandb
from .logger import Logger


class WandbLogger(Logger):
    def __init__(self, run_name: str, config: dict | None = None, project: str = "AD-Project", entity: str = "federated-flower-wanb"):
        self.run = wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            reinit=True,
            resume="allow",
            config=config,
        )
        self.declared_metrics = set()

        # Declare all required metrics with their associated step metric
        self.define_metric("centralized/test_loss", step_metric="centralized/fed-round")
        self.define_metric("centralized/test_accuracy", step_metric="centralized/fed-round")
        self.define_metric("centralized/client_avg_train_loss", step_metric="centralized/fed-round")
        self.define_metric("centralized/client_avg_val_loss", step_metric="centralized/fed-round")
        self.define_metric("centralized/client_avg_val_accuracy", step_metric="centralized/fed-round")
        self.define_metric("centralized/client_avg_test_loss", step_metric="centralized/fed-round")
        self.define_metric("centralized/client_avg_test_accuracy", step_metric="centralized/fed-round")

    def define_metric(self, metric: str, step_metric: str = None):
        """
        Defines a WandB metric with an optional step metric, if not already defined.
        """
        if metric not in self.declared_metrics:
            if step_metric:
                wandb.define_metric(metric, step_metric=step_metric)
            else:
                wandb.define_metric(metric)
            self.declared_metrics.add(metric)

    def log(self, data: dict, name: str = '', step: int | float = None):
        """
        Logs a dictionary of data to Weights & Biases, with optional namespacing and step.
        Automatically defines metrics if not already declared.
        """
        log_data = {}
        for key, value in data.items():
            full_key = f"{name}/{key}" if name else key
            log_data[full_key] = value
            self.define_metric(full_key)  # Dynamically define if new metric is used

        wandb.log(log_data, step=step)

    def finish(self):
        """
        Properly finishes the WandB run.
        """
        wandb.finish()