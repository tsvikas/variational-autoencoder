#!/usr/bin/bash
chmod -x -R .
chmod +x src/train.py ./configure.sh
sudo apt update
sudo apt install -y python3.11
curl -sSL https://install.python-poetry.org | python3 -
poetry env use python3.11
poetry install
poetry run wandb login "$(cat wandb_api_key.secret)"
poetry run jupytext --sync src/explore_model.py
echo PYTHON_BIN = "$(poetry env info -p)"/bin/python
echo CMD = poetry run src/train.py
echo JUPYTER = poetry run jupyter lab --no-browser --ip 0.0.0.0
grep -F .token ~/.jupyter/jupyter_config.py
