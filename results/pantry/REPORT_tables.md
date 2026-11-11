## Closed-loop query success by interference level k (cross-session trials)

**Budget B=2**

| system | k=0 | k=1 | k=3 | k=7 | mean | single-task mean | frame frac |
|---|---|---|---|---|---|---|---|
| none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | nan |
| random | 0.30 | 0.20 | 0.08 | 0.03 | 0.15 | 0.55 | 0.00 |
| recency_frame | 0.14 | 0.00 | 0.00 | 0.01 | 0.04 | 0.54 | 1.00 |
| recency_sym | 0.21 | 0.02 | 0.00 | 0.03 | 0.07 | 0.53 | 0.00 |
| scripted_mem_dropfail | 0.56 | 0.17 | 0.03 | 0.06 | 0.20 | 0.49 | nan |
| scripted_mem | 0.71 | 0.13 | 0.03 | 0.04 | 0.23 | 0.54 | nan |
| learned_online_sym | 0.14 | 0.06 | 0.06 | 0.10 | 0.09 | 0.65 | 0.00 |
| learned_sleep_sym | 0.16 | 0.19 | 0.09 | 0.02 | 0.11 | 0.40 | 0.00 |
| learned_both_sym | 0.12 | 0.07 | 0.03 | 0.09 | 0.08 | 0.65 | 0.00 |
| learned_split_sym | 0.13 | 0.26 | 0.07 | 0.14 | 0.15 | 0.50 | 0.00 |
| oracle | 0.81 | 0.83 | 0.78 | 0.77 | 0.80 | 0.53 | 0.00 |

**Budget B=4**

| system | k=0 | k=1 | k=3 | k=7 | mean | single-task mean | frame frac |
|---|---|---|---|---|---|---|---|
| none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | nan |
| random | 0.66 | 0.37 | 0.20 | 0.03 | 0.31 | 0.85 | 0.00 |
| recency_frame | 0.41 | 0.02 | 0.00 | 0.02 | 0.11 | 0.80 | 1.00 |
| recency_sym | 0.88 | 0.03 | 0.04 | 0.03 | 0.25 | 0.99 | 0.00 |
| scripted_mem_dropfail | 0.83 | 0.48 | 0.10 | 0.04 | 0.36 | 0.94 | nan |
| scripted_mem | 0.99 | 0.49 | 0.10 | 0.06 | 0.41 | 0.99 | nan |
| learned_sleep | 0.26 | 0.23 | 0.14 | 0.14 | 0.19 | 0.40 | 0.40 |
| learned_both | 0.47 | 0.21 | 0.06 | 0.03 | 0.19 | 0.80 | 0.69 |
| learned_online_sym | 0.80 | 0.32 | 0.03 | 0.04 | 0.30 | 0.99 | 0.00 |
| learned_sleep_sym | 0.22 | 0.23 | 0.11 | 0.07 | 0.16 | 0.40 | 0.00 |
| learned_both_sym | 0.53 | 0.21 | 0.03 | 0.06 | 0.21 | 0.99 | 0.00 |
| learned_split_sym | 0.51 | 0.47 | 0.38 | 0.44 | 0.45 | 0.60 | 0.00 |
| oracle | 0.98 | 0.99 | 0.97 | 1.00 | 0.98 | 0.99 | 0.00 |

**Budget B=8**

| system | k=0 | k=1 | k=3 | k=7 | mean | single-task mean | frame frac |
|---|---|---|---|---|---|---|---|
| none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | nan |
| random | 0.94 | 0.67 | 0.34 | 0.21 | 0.54 | 0.99 | 0.00 |
| recency_frame | 0.59 | 0.07 | 0.01 | 0.00 | 0.17 | 0.79 | 1.00 |
| recency_sym | 0.94 | 0.36 | 0.00 | 0.00 | 0.33 | 0.99 | 0.00 |
| scripted_mem_dropfail | 0.83 | 0.90 | 0.68 | 0.22 | 0.66 | 0.94 | nan |
| scripted_mem | 0.99 | 0.99 | 0.76 | 0.22 | 0.74 | 0.99 | 0.00 |
| learned_online_sym | 0.98 | 0.76 | 0.42 | 0.27 | 0.61 | 0.99 | 0.00 |
| learned_sleep_sym | 0.29 | 0.31 | 0.12 | 0.13 | 0.21 | 0.40 | 0.00 |
| learned_both_sym | 0.76 | 0.66 | 0.57 | 0.48 | 0.61 | 0.99 | 0.00 |
| learned_split_sym | 0.96 | 0.86 | 0.79 | 0.71 | 0.83 | 0.99 | 0.00 |
| oracle | 0.94 | 0.87 | 0.79 | 1.00 | 0.90 | 0.99 | 0.00 |

## Single-session success (largest budget)

| system | find | count | sequence | adapt | tidy |
|---|---|---|---|---|---|
| none | 0.29 | 0.96 | 0.00 | 0.75 | 0.00 |
| random | 0.98 | 0.96 | 1.00 | 1.00 | 1.00 |
| recency_frame | 0.98 | 0.96 | 0.02 | 1.00 | 1.00 |
| recency_sym | 0.98 | 0.96 | 1.00 | 1.00 | 1.00 |
| scripted_mem_dropfail | 0.98 | 0.96 | 1.00 | 0.75 | 1.00 |
| scripted_mem | 0.98 | 0.96 | 1.00 | 1.00 | 1.00 |
| learned_online_sym | 0.98 | 0.96 | 1.00 | 1.00 | 1.00 |
| learned_sleep_sym | 0.29 | 0.96 | 0.00 | 0.75 | 0.00 |
| learned_both_sym | 0.98 | 0.96 | 1.00 | 1.00 | 1.00 |
| learned_split_sym | 0.98 | 0.96 | 1.00 | 1.00 | 1.00 |
| oracle | 0.98 | 0.96 | 1.00 | 1.00 | 1.00 |

## Policy-free decision distortion: expert agreement given only the memory (query steps)

**B=2**

| system | k=0 | k=1 | k=3 | k=7 | within-session |
|---|---|---|---|---|---|
| none | 0.48 | 0.45 | 0.44 | 0.44 | 0.56 |
| random | 0.78 | 0.60 | 0.50 | 0.47 | 0.66 |
| recency_frame | 0.59 | 0.45 | 0.44 | 0.44 | 0.79 |
| recency_sym | 0.59 | 0.45 | 0.44 | 0.44 | 0.79 |
| scripted_mem_dropfail | 0.84 | 0.51 | 0.44 | 0.44 | 0.82 |
| scripted_mem | 0.84 | 0.50 | 0.44 | 0.44 | 0.83 |
| learned_online_sym | 0.67 | 0.45 | 0.44 | 0.44 | 0.90 |
| learned_sleep_sym | 0.87 | 0.78 | 0.64 | 0.59 | 0.56 |
| learned_both_sym | 0.64 | 0.48 | 0.44 | 0.44 | 0.90 |
| learned_split_sym | 0.67 | 0.65 | 0.62 | 0.64 | 0.65 |
| oracle | 0.87 | 1.00 | 1.00 | 1.00 | 0.66 |

**B=4**

| system | k=0 | k=1 | k=3 | k=7 | within-session |
|---|---|---|---|---|---|
| none | 0.48 | 0.45 | 0.44 | 0.44 | 0.56 |
| random | 0.94 | 0.70 | 0.60 | 0.49 | 0.77 |
| recency_frame | 0.93 | 0.45 | 0.44 | 0.44 | 1.00 |
| recency_sym | 0.93 | 0.45 | 0.44 | 0.44 | 1.00 |
| scripted_mem_dropfail | 1.00 | 0.73 | 0.48 | 0.44 | 0.99 |
| scripted_mem | 1.00 | 0.71 | 0.47 | 0.44 | 1.00 |
| learned_sleep | 1.00 | 1.00 | 1.00 | 0.99 | 0.56 |
| learned_both | 0.88 | 0.62 | 0.46 | 0.44 | 0.93 |
| learned_online_sym | 1.00 | 0.63 | 0.46 | 0.44 | 1.00 |
| learned_sleep_sym | 1.00 | 0.87 | 0.72 | 0.65 | 0.56 |
| learned_both_sym | 0.87 | 0.60 | 0.45 | 0.44 | 1.00 |
| learned_split_sym | 0.87 | 0.88 | 0.85 | 0.80 | 0.85 |
| oracle | 1.00 | 1.00 | 1.00 | 1.00 | 0.72 |

**B=8**

| system | k=0 | k=1 | k=3 | k=7 | within-session |
|---|---|---|---|---|---|
| none | 0.48 | 0.45 | 0.44 | 0.44 | 0.56 |
| random | 1.00 | 0.96 | 0.74 | 0.55 | 0.83 |
| recency_frame | 1.00 | 0.70 | 0.44 | 0.44 | 1.00 |
| recency_sym | 1.00 | 0.70 | 0.44 | 0.44 | 1.00 |
| scripted_mem_dropfail | 1.00 | 1.00 | 0.86 | 0.50 | 0.99 |
| scripted_mem | 1.00 | 1.00 | 0.86 | 0.50 | 1.00 |
| learned_online_sym | 1.00 | 0.93 | 0.76 | 0.56 | 1.00 |
| learned_sleep_sym | 1.00 | 1.00 | 0.95 | 0.78 | 0.56 |
| learned_both_sym | 0.99 | 0.81 | 0.71 | 0.70 | 1.00 |
| learned_split_sym | 1.00 | 0.91 | 0.87 | 0.82 | 1.00 |
| oracle | 1.00 | 1.00 | 1.00 | 1.00 | 0.76 |

## What the memories hold at query time (event-type histogram, cross trials)

- random B=2: drawer 41%, session 15%, picked 9%, strat_fail 8%, delivered 6%, cue_seq 5%
- recency_frame B=2: drawer/frame 32%, picked/frame 25%, delivered/frame 16%, pressed/frame 11%, placed_bin/frame 8%, cue_seq/frame 4%
- recency_sym B=2: drawer 32%, picked 25%, delivered 16%, pressed 10%, placed_bin 8%, cue_seq 5%
- scripted_mem_dropfail B=2: drawer 75%, strat_ok 16%, drawer_empty 9%
- scripted_mem B=2: drawer 70%, strat_ok 20%, drawer_empty 9%, strat_fail 2%
- learned_online_sym B=2: drawer 38%, delivered 16%, placed_bin 10%, strat_fail 8%, cue_seq 8%, pressed 8%
- learned_sleep_sym B=2: drawer 81%, strat_fail 9%, drawer_empty 6%, strat_ok 2%, pressed 1%
- learned_both_sym B=2: session 40%, drawer 22%, cue_seq 13%, strat_fail 6%, placed_bin 6%, strat_ok 5%
- learned_split_sym B=2: drawer 65%, strat_fail 27%, drawer_empty 8%
- oracle B=2: drawer 68%, strat_ok 13%, picked 13%, drawer_empty 5%
- random B=4: drawer 38%, session 18%, picked 10%, strat_fail 8%, cue_seq 6%, delivered 6%
- recency_frame B=4: drawer/frame 35%, picked/frame 16%, cue_seq/frame 10%, delivered/frame 9%, placed_bin/frame 7%, pressed/frame 6%
- recency_sym B=4: drawer 35%, picked 16%, pressed 12%, delivered 9%, placed_bin 7%, strat_ok 5%
- scripted_mem_dropfail B=4: drawer 83%, strat_ok 9%, drawer_empty 7%
- scripted_mem B=4: drawer 79%, strat_ok 13%, drawer_empty 7%, strat_fail 1%
- learned_sleep B=4: drawer 57%, drawer/frame 16%, drawer_empty/frame 6%, strat_fail/frame 5%, strat_ok/frame 4%, picked/frame 4%
- learned_both B=4: drawer 31%, session 20%, cue_seq 6%, strat_fail 5%, drawer/frame 5%, picked 5%
- learned_online_sym B=4: drawer 46%, cue_seq 14%, delivered 9%, strat_fail 8%, strat_ok 6%, placed_bin 5%
- learned_sleep_sym B=4: drawer 81%, strat_fail 8%, drawer_empty 6%, strat_ok 1%, pressed 1%, picked 1%
- learned_both_sym B=4: drawer 55%, cue_seq 13%, strat_fail 11%, pressed 5%, strat_ok 5%, drawer_empty 5%
- learned_split_sym B=4: drawer 66%, strat_fail 19%, strat_ok 8%, drawer_empty 7%
- oracle B=4: drawer 66%, picked 8%, strat_ok 7%, strat_fail 7%, delivered 7%, drawer_empty 6%
- random B=8: drawer 35%, session 18%, picked 11%, delivered 7%, strat_fail 7%, cue_seq 6%
- recency_frame B=8: drawer/frame 31%, picked/frame 14%, session/frame 14%, delivered/frame 9%, cue_seq/frame 9%, pressed/frame 6%
- recency_sym B=8: drawer 30%, session 14%, picked 13%, pressed 10%, delivered 9%, cue_seq 7%
- scripted_mem_dropfail B=8: drawer 83%, strat_ok 10%, drawer_empty 7%
- scripted_mem B=8: drawer 76%, strat_ok 12%, drawer_empty 7%, strat_fail 4%
- learned_online_sym B=8: drawer 41%, session 10%, cue_seq 9%, strat_fail 7%, strat_ok 7%, pressed 6%
- learned_sleep_sym B=8: drawer 72%, strat_fail 11%, drawer_empty 5%, session 3%, strat_ok 2%, pressed 2%
- learned_both_sym B=8: drawer 50%, strat_ok 10%, pressed 7%, picked 6%, strat_fail 6%, cue_seq 5%
- learned_split_sym B=8: drawer 77%, strat_ok 15%, strat_fail 4%, pressed 2%, drawer_empty 1%, picked 1%
- oracle B=8: drawer 42%, session 24%, picked 8%, strat_fail 7%, delivered 5%, strat_ok 5%
