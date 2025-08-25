# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# sgd

cd ..

for split in train
do
    for template in no_template # llama2 llama3
    do  
        for all_turn in False #True
        do
            python -m src.sgd.prompting \
                                                    --split $split \
                                                    --template $template \
                                                    --all_turn $all_turn
        done
    done
done
