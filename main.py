import argparse
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TestTubeLogger
from serde.yaml import from_yaml
from channel_order.app import TrainSystem
from channel_order.config import Config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def run():
    args = get_args()
    config = from_yaml(Config, Path(args.config).read_text())
    logger = TestTubeLogger(
        save_dir=str(config.save_dir),
        name=config.experiment_name,
        version=config.version,
    )
    app = TrainSystem(config)
    trainer = Trainer(
        min_epochs=50,
        max_epochs=config.epochs,
        auto_lr_find=True,
        auto_scale_batch_size=True,
        logger=logger,
        auto_select_gpus=True,
        gpus=[0],
        num_processes=2,
        precision=16,
        callbacks=[EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(app)
    trainer.test(app)


if __name__ == "__main__":
    run()
