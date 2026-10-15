#!/bin/bash
# télécharge le dataset FreSaDa (Ionescu & Chifu, 2021) dans data/
# le dataset est sous licence CC BY-NC-SA 4.0 donc on le met pas directement dans le repo
mkdir -p data
if [ ! -d data/FreSaDa ]; then
  git clone --depth 1 https://github.com/adrianchifu/FreSaDa.git data/FreSaDa
fi
echo "ok : données dans data/FreSaDa"
