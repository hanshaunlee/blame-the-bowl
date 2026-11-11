# usage: POLICY=ckpt/policy_v4.pt MODE=both B=4 ; runs writer RL then nothing else (eval is a separate sweep)
set -ex
python -m hm.train.writer_rl --policy $POLICY --mode $MODE --B $B --updates 500 --G 8 --trials_per_update 16 --out ckpt/writer_${MODE}_B${B}.pt
