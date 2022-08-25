# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
import neptune.new as neptune

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    run = neptune.init(
        project="ksh/MonoSVO",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMmI1MDI3Ny0zZDIwLTRhOGMtOTQ2OS1kOGJiMTVlNDNjM2IifQ==",
        source_files=[
            "trainer.py",
            "networks/[A-Za-z]*.py",
            "datasets/[A-Za-z]*.py"
        ],
        tags=["monodepth2"]
    )

    params = {
        "model_name": opts.model_name,
        "split": opts.split,
        "dataset": opts.dataset,
        "learning_rate": opts.learning_rate,
        "scheduler_step_size": opts.scheduler_step_size,
        "batch_size": opts.batch_size,
        "num_epochs": opts.num_epochs,
        "pose_model_type": opts.pose_model_type,
        "deformable_conv": opts.deformable_conv,
        "uncertainty_input": opts.uncertainty_input}
    run["parameters"] = params

    trainer = Trainer(opts)
    trainer.train()
