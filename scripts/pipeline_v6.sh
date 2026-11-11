set -ex
# fresh data under the current dynamics (jam, memory-only queries), fine-tune v5 -> v6
python -m hm.train.bc --init ckpt/policy_v5.pt --n_train 30000 --data data/trials_train_v6.pkl --val data/trials_val_v6.pkl --epochs 3 --lr 1e-4 --workers 14 --out ckpt/policy_v6.pt
# one DAgger round on the new dynamics
python -m hm.train.dagger --policy ckpt/policy_v6.pt --n 3000 --seed 21 --out data/dagger_v6.pkl
python -m hm.train.bc --init ckpt/policy_v6.pt --data data/trials_train_v6.pkl --val data/trials_val_v6.pkl --extra_data data/dagger_v6.pkl --epochs 2 --lr 5e-5 --workers 14 --out ckpt/policy_v6d.pt
python -m hm.eval.sweep --policy ckpt/policy_v6d.pt --n 240 --budgets 4 --systems none recency_sym scripted_mem oracle --out results/check_v6d.json
