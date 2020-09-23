export PYTHONDONTWRITEBYTECODE=1

# MAML miniimagenet
python -u run_miniimagenet_maml.py --shots 1 --meta-shots 15 --inner-iters 5 --meta-step 0.001 --meta-batch 4 --meta-iters 60000 --eval-iters 10 --learning-rate 0.01 --eval-samples 600 --order 2 | tee output_miniimagenet_maml_order2_seed0.txt

# MAML omniglot
# setting meta-shots=15 and inner-iters=5 did not significantly change the performance. Consider validating.
python -u run_omniglot_maml.py --shots 1 --meta-shots 1 --inner-iters 1 --meta-step 0.001 --meta-batch 32 --meta-iters 60000 --eval-iters 10 --learning-rate 0.4 --eval-samples 600 --order 2 | tee output_omniglot_maml_order2_seed0.txt

# REPTILE miniimagenet
python -u run_miniimagenet_reptile.py --seed 0 --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 5 --eval-iters 50 --eval-samples 1000 --learning-rate 0.001 --train-shots 15 | tee reptile_miniimagenet_seed0.txt

# REPTILE omniglot
python -u run_omniglot_reptile.py --seed 0 --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --eval-samples 1000 --learning-rate 0.001 --train-shots 10 | tee reptile_omniglot_seed0.txt


