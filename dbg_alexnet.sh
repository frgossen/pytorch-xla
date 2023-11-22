






# # Rebuild incl C/C++ code
# cd xla && XLA_CUDA=1 python setup.py develop && cd .. 




export PJRT_DEVICE=CUDA

python xla/benchmarks/experiment_runner.py \
    --suite-name="torchbench" \
    --experiment-config="{\"accelerator\": \"cuda\", \"xla\": \"PJRT\", \"xla_flags\": null, \"dynamo\": \"openxla_eval\", \"test\": \"eval\"}" "--model-config={\"model_name\": \"alexnet\"}" \
    --repeat 5 \
    --profile-cuda 


