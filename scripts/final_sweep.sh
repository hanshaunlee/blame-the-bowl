set -ex
POLICY=${POLICY:-ckpt/policy_v6d.pt}
ls ckpt/writer_*_v6*.pt
python -m hm.eval.sweep --policy $POLICY --n 600 --budgets 2 4 8 --learned ckpt/writer_*_v6*.pt --out results/sweep_final.json
python -m hm.eval.distortion --policy $POLICY --n 600 --budgets 2 4 8 --learned ckpt/writer_*_v6*.pt --out results/distortion_final.json
