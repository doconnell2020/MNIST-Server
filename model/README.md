# MNIST Model Package

## Overview
This subpackage serves as the model training and saving for the MNIST Server application.

## Development
This package is tracked using `uv workspaces`. Consider using the same tool and availing of the `uv sync` utility.

When developing locally, the model runs will be saved in a local folder named `mlruns`
in the root of this package.

Use the `mlflow ui` command to initialize a local MLFlow server to view these runs in a friendly UI.


## Future work

Environment dependent configurations will be a roadmap item to account for local, review (PR/MR),
and production deployments.

Additionally, the `main.py` script could become more modular to allow for other model types to be called on.
