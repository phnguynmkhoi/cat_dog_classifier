#!/bin/bash
curl -L -o ./data/data.zip\
  https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset

unzip -o ./data/data.zip -d ./data
rm ./data/data.zip