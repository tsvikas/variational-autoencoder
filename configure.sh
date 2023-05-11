#!/usr/bin/bash
sudo apt update
sudo apt install -y python3.11
curl -sSL https://install.python-poetry.org | python3 -
poetry env use python3.11
poetry install
poetry run wandb login
echo "$(poetry env info -p)"/bin/python
