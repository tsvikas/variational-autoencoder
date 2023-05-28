#!/usr/bin/bash
# set executeable files
if [ -d ".git" ]
then
  git ls-files | xargs chmod -x
else
  find . -type f -executable -exec chmod -x {} +
fi
chmod +x src/train.py ./configure.sh
# apt install packages
sudo apt update -qq
sudo apt install -qqy python3.11 dos2unix
# set correct line ending
if [ -d ".git" ]
then
  git ls-files | xargs dos2unix
else
  find . -type f -exec dos2unix {} +
fi
# install poetry and venv
curl -sSL https://install.python-poetry.org | python3 -
poetry env use python3.11
poetry install
# setup needed libraries
poetry run wandb login "$(cat wandb_api_key.secret)"
poetry run jupytext --to ipynb src/explore_model.py
echo "set new password for the jupyter server"
poetry run jupyter lab password
# print info to screen
echo
echo PYTHON_BIN = "$(poetry env info -p)"/bin/python
echo TRAIN_CMD = poetry run src/train.py
echo JUPYTER_CMD = poetry run jupyter lab --no-browser --ip 0.0.0.0 --notebook-dir src
echo TOKEN = "$(grep -F .token ~/.jupyter/jupyter_config.py | cut -d\' -f2)"
