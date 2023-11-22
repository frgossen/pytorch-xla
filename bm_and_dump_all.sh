






# # Rebuild incl C/C++ code
# cd xla && XLA_CUDA=1 python setup.py develop && cd .. 





python xla/benchmarks/experiment_runner.py \
    --suite-name="torchbench" \
    --accelerator=cuda \
    --xla=PJRT --xla=None \
    --dynamo=openxla_eval --dynamo=openxla --dynamo=inductor \
    --test=eval --test=train \
    --repeat 5 \
    --profile-cuda \
    --filter="alexnet|BERT_pytorch|hf_T5|basic_gnn_gcn"




