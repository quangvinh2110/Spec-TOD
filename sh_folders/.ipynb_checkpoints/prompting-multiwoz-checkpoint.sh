# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# multiwoz
cd ..

for dataset_version in 2.2
do
    for split in train
    do
        for n_sample in 845 # 84 #400 800
        do
            for template in llama3 # vicuna
            do  
                for all_turn in False #True
                do 
                    for task in e2e # dst
                    do 
                        python -m src.multiwoz.prompting \
                            --dataset_version $dataset_version \
                            --split $split \
                            --n_sample $n_sample \
                            --template $template \
                            --all_turn $all_turn \
                            --task $task
                    done
                done
            done
        done
    done
done