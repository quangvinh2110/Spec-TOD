# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

cd ..

python3 -m src.sgd.preprocess
cd ./data/pre-training_corpora/seperate_datasets
mkdir Schema_Guided
cd ../../..
mv ./src/sgd/normalized_schema.yml ./data/pre-training_corpora/seperate_datasets/Schema_Guided/
python3 -m src.sgd.postprocess
