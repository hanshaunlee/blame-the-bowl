# Remember the Decision: learned memory compression for long-horizon robot policies

*2026-11-04. Numbers from `results/sweep_final.json` (closed-loop, 600 trials, student v6d) and
`results/distortion_final.json` (policy-free, 600 trials). Figures: `results/sweep_final_query.png`,
`results/sweep_final_single.png`, `results/distortion_final.png`. Full tables: `results/REPORT_tables.md`.*

## 1. Question

A robot policy with a budgeted long-term memory must decide *what to keep* and *when to decide*.
MEM (Torné et al., 2026) answers both by hand: an LLM prompt compresses the language memory
online as events arrive, and it drops failed attempts. We ask whether the compression policy can
be learned from the downstream decision instead, whether deciding in hindsight at session
boundaries beats deciding online, and what a decision-trained memory chooses to keep.

## 2. Setup

**Pantry.** A 9x9 pixel world (64x64 RGB, top-down) with drawers that auto-close (state
aliasing), an opaque bin, a cue panel that blanks after three steps, four buttons, and a cabinet
with a hidden latch that jams for two steps after a wrong strategy. Tasks follow the RoboMME
taxonomy: find (spatial memory under occlusion), count (temporal), sequence (procedural), adapt
(in-context adaptation after failed strategies), tidy. Cross-session trials follow
RoboMME-Interference: session 0 reveals facts about layout L0, then k in {0,1,3,7} distractor
sessions on other layouts, then a query on L0 (find or adapt) with a step limit chosen so that the
query is solvable only from memory (the no-memory expert cannot finish in time).

**Memory units.** Every step emits typed events (drawer i holds colour c, cue says k, strategy s
failed, ...) and a frame. A memory unit is either the event vector or the frame, mirroring MEM's
text and video tracks. Long-term memory is B slots.

**Belief-state expert.** The expert acts on the history of events it has observed, never on
hidden state, so its behaviour is a function of history and "what to remember" has a ground
truth.

**Information-matched student.** A 2.2M-parameter transformer over [4 frames, context, 13
fixture tokens, B memory slots]. Its imitation label at each step is the expert's action computed
from exactly the information the student is given (sampled memory subset plus the three-frame
working window). Trained on 30k trials plus one DAgger round: 98.9% per-step agreement; with
oracle memory it solves 97 to 100% of cross-session queries and 100% of single-session find,
sequence, tidy and adapt (count is capped at 96% by the step limit).

**Memory systems.** none; recency (facts); recency (frames, i.e. keyframes at every event);
reservoir random; a scripted MEM-style compressor (latest fact per drawer, drop a drawer fact when
its object is taken, keep the latest progress fact, drop failed strategies once the latch is
known, drop session-local facts at session end, FIFO at the budget), with and without MEM's
"drop failed attempts" rule; a cheating oracle that protects the facts the future query needs;
and four learned systems trained with GRPO against the frozen student with reward equal to the
student's log-probability of the full-information expert's action on memory-relevant steps:
*online* (at each event: overwrite slot j, or drop), *sleep* (re-select B items at session end
from old slots and the session's events, without seeing the next task), *both* (online then
re-select), and *split* (B/2 online slots cleared at session end plus B/2 consolidated slots, a
two-scale memory in the spirit of MEM).

**Metrics.** Closed-loop success per task and per interference level k. Policy-free decision
distortion: the fraction of memory-relevant steps on expert trajectories at which the expert,
given only the memory contents, takes the same action as the full-history expert.

## 3. Theory that does not depend on the sandbox

1. *Decision sufficiency.* The only thing imitation asks of memory is that the expert's action be
   recoverable from it. Any objective not monotone in decision distortion (reconstruction,
   coverage, recency, similarity) can be arbitrarily bad at the same budget (DeMem's argument,
   transferred from bandits to sequential imitation).
2. *Hindsight gap.* If a session emits n facts of which an unpredictable s are superseded later in
   the session (object taken, strategy superseded, progress counts), a causal writer with budget
   B retains at most B(1 - s/n) still-valid facts in expectation, while a session-end
   consolidator retains min(B, n - s). The gap is a property of the write schedule, not of the
   student.
3. *Interference collapse.* With k distractor sessions of m facts and a FIFO store of size B, the
   relevant fact survives iff k m < B; success is a step in k. Any relevance-aware writer is flat.

## 4. Results (student v6d, 600 trials, 90 cross trials per k)

### 4.1 Heuristics and oracle: closed-loop query success by k

| system | B=2: k0 k1 k3 k7 | B=4: k0 k1 k3 k7 | B=8: k0 k1 k3 k7 |
|---|---|---|---|
| none | .00 .00 .00 .00 | .00 .00 .00 .00 | .00 .00 .00 .00 |
| random | .30 .20 .08 .03 | .66 .37 .20 .03 | .94 .67 .34 .22 |
| recency (frames) | .14 .00 .00 .01 | .41 .02 .00 .02 | .59 .07 .01 .00 |
| recency (facts) | .21 .02 .00 .03 | .88 .03 .04 .03 | .94 .36 .00 .00 |
| scripted MEM-style, drop failures | .56 .17 .03 .06 | .83 .48 .10 .04 | .83 .90 .68 .22 |
| scripted MEM-style, keep failures | .71 .13 .03 .04 | .99 .49 .10 .06 | .99 .99 .76 .22 |
| oracle | .81 .83 .78 .77 | .98 .99 .97 1.0 | .94 .87 .79 1.0 |

Observations. (i) The interference step predicted in Section 3.3: recency holds at k=0 and is at
the no-memory level from k=1. (ii) The scripted compressor is as good as the oracle at k=0 for
B>=4 and decays to 22% at k=7 even with B=8. (iii) Frames are a poor procedural memory: recency
(frames) solves 2% of sequence tasks where recency (facts) solves 100%, consistent with RoboMME.
(iv) MEM's "drop failed attempts" rule costs 25 points on adapt (0.75 vs 1.00 single-session).

### 4.2 Learned systems (symbolic units, GRPO against the frozen student)

Closed-loop query success by k:

| system | B=2 | B=4 | B=8 |
|---|---|---|---|
| online writer | .14 .06 .06 .10 | .80 .32 .03 .04 | .98 .76 .42 .27 |
| hindsight consolidator (sleep) | .16 .19 .09 .02 | .22 .23 .11 .07 | .29 .31 .12 .13 |
| online + consolidate (both) | .12 .07 .03 .09 | .53 .21 .03 .06 | .76 .66 .57 .48 |
| two-scale (split: B/2 online, B/2 consolidated) | .13 .26 .07 .14 | .51 .47 .38 .44 | .96 .86 .79 .71 |
| scripted MEM-style (reference) | .71 .13 .03 .04 | .99 .49 .10 .06 | .99 .99 .76 .22 |
| oracle (reference) | .81 .83 .78 .77 | .98 .99 .97 1.0 | .94 .87 .79 1.0 |

Findings. (i) **Under interference the learned two-scale memory is the only non-oracle system
that holds**: at B=8, k=7 it keeps 71% versus 22% for the scripted compressor, 21% for random
and 0% for recency; at k<=3 it tracks the oracle (.96/.86/.79 vs .94/.87/.79). (ii) **The online
writer collapses exactly like recency** (B=4: .80 -> .03 by k=3), confirming the write-time
argument of Section 3.2 at scale of this benchmark. (iii) **The pure hindsight consolidator is
the best selector but the worst closed-loop system**: it has no within-session memory, so the
student cannot execute session 0 well (within-session agreement 0.56) and the facts it would
later need are never generated. This is a learned justification for MEM's two-scale design,
short-horizon working memory plus consolidated long-term memory. (iv) At B=4 all learned systems
are under-trained relative to the oracle (see 4.3); the recipe works but needs more RL budget at
small B.

### 4.3 Policy-free decision distortion (expert agreement on query steps, k = 0 1 3 7)

| system | B=2 | B=4 | B=8 |
|---|---|---|---|
| none | .48 .45 .44 .44 | .48 .45 .44 .44 | .48 .45 .44 .44 |
| random | .78 .60 .50 .47 | .94 .70 .60 .49 | 1.0 .96 .74 .55 |
| recency (facts) | .59 .45 .44 .44 | .93 .45 .44 .44 | 1.0 .70 .44 .44 |
| scripted MEM-style | .84 .50 .44 .44 | 1.0 .71 .47 .44 | 1.0 1.0 .86 .50 |
| online writer | .67 .45 .44 .44 | 1.0 .63 .46 .44 | 1.0 .93 .76 .56 |
| hindsight consolidator | .87 .78 .64 .59 | 1.0 .87 .72 .65 | 1.0 1.0 .95 .78 |
| two-scale (split) | .67 .65 .62 .64 | .87 .88 .85 .80 | 1.0 .91 .87 .82 |
| hindsight consolidator, frames allowed | - | 1.0 1.0 1.0 .99 | - |
| oracle | .87 1.0 1.0 1.0 | 1.0 1.0 1.0 1.0 | 1.0 1.0 1.0 1.0 |

Stripped of the student, the ordering is clean: consolidate > split > online = scripted = recency
under interference, at every budget. With frames allowed the consolidator reaches the oracle
(B=4, 0.99 at k=7) but the student reads frames worse than facts (96% vs 87% query-step accuracy),
which is why the closed-loop runs use symbolic units.

## 5. What the memories hold

At query time the scripted compressor holds drawer facts (70%) and the known latch (20%). The
learned consolidator holds drawer facts (81%) and the failed strategies (9%), i.e. it keeps the
lesson of a failed attempt rather than dropping it; the online writer's slots are 38% drawer
facts and the rest session-local progress (delivered, placed, pressed) that is useless across
sessions; "both" fills 40% of slots with session markers, the clearest sign that its online
half overwrites what its consolidator kept.

## 6. Limitations and the path to scale

Pantry is a mechanism study: pixels and a manipulation-shaped task taxonomy, but a gridworld.
The claims that transfer and how to test them on RoboMME with the released π0.5 memory variants
are in `results/SCALE_PLAN.md`.
