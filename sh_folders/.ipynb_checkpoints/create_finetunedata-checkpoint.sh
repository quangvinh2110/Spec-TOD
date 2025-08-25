# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

cd ..

for pd_size in 200  #422 84 #100 400
do
    python3 create_finetunedata.py --configfile ./data/finetunedata/sft-llama3.yml \
                        --outputfile ./data/finetunedata/sft-llama3-pd-v1-$pd_size.json \
                        --domain_size $pd_size \
                        --max_len 4096
done
