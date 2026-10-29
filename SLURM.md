# Slurm on Iridis X: DeAI cheat sheet

Commands for running this project's training jobs on Iridis X. Replace `<jobid>`, `<run>` and `<node>` with your own values; the job ID is printed by `sbatch` and listed by `squeue`.

## One-time setup (login node)

The login node has internet and the GPU nodes don't, so everything a job needs is fetched here first.

```bash
module avail python                        # find the Python 3.12 module
module load python/3.12.6
scripts/hpc_setup.sh --base Qwen/Qwen3.5-9B   # venv, data, models, kernels, hpc_env.sh
```

Optional, to build the `causal-conv1d` kernel, which needs a CUDA 12.x toolkit to match torch's CUDA 12.8:

```bash
module avail cuda                          # the default (13.x) won't work
module load cuda/12.8                      # use the name module avail shows
scripts/hpc_setup.sh --base Qwen/Qwen3.5-9B   # rerun: finished steps are skipped
```

## Cluster status

```bash
sinfo -s                                   # all partitions, nodes idle/used
sinfo -p a100                              # a100 nodes and their states
sinfo -p a100 -N -O NodeList:12,StateLong:12,Gres:30,GresUsed:40   # GPUs used per node
```

Node states:

| State | Meaning |
|---|---|
| `idle` | Free |
| `mixed` | Partly in use; check `GresUsed` to see if its GPUs are free |
| `allocated` | Full |
| trailing `-` (e.g. `mixed-`) | Planned for a pending job |

## Submitting

```bash
# Smoke test first (short time limit starts sooner via backfill)
DEAI_RUN=smoke-9b sbatch --time=01:00:00 slurm_train_a100x2_qwen35_9b.sh \
    --rounds 1 --train-limit 8 --eval-limit 8 --n 4

# Full run, one job per reward source
sbatch slurm_train_a100x2_qwen35_9b.sh                      # quick reward
for r in quick neural-quick neural stat full; do
    DEAI_REWARD=$r sbatch slurm_train_a100x2_qwen35_9b.sh
done

# With email when it ends or fails
sbatch --mail-type=END,FAIL --mail-user=<email> slurm_train_a100x2_qwen35_9b.sh
```

- **Options on the command line** (`--time`, `--gres`, `--mem`, `-p`, ...) override the script's `#SBATCH` lines.
- **Settings:** `DEAI_*` variables set before `sbatch` choose the run name, reward, base and other trainer settings.
- **Arguments after the script name** go to `deai.train_dpo`.
- **Resuming:** requeue is disabled on Iridis X, so if a job stops, resubmit the same command; finished stages are skipped.

| Script | Hardware | Use |
|---|---|---|
| `slurm_train_a100x2_qwen35_9b.sh` | 2× A100 | Qwen3.5-9B |
| `slurm_train_a100.sh` | 1× A100 | Qwen3-4B |
| `slurm_train_l4_multi.sh` | 7× L4 | One Qwen3-4B run across all of them |
| `slurm_train_l4_array.sh` | 1× L4 per task | One Qwen3-4B run per reward source (job array) |

## Watching a job

```bash
squeue -u $USER                            # your jobs; ST: PD = pending, R = running
squeue -j <jobid> --start                  # estimated start time of a pending job
scontrol show job <jobid>                  # everything about a job (node, limits, reason)
```

Common pending reasons in the `NODELIST(REASON)` column:

| Reason | Meaning |
|---|---|
| `Resources` | Waiting for free GPUs; it will start on its own |
| `Priority` | Other jobs are ahead in the queue |
| `QOSMaxJobsPerUserLimit` | You already have the maximum number of jobs running |

Logs, which keep being written after you disconnect:

```bash
tail -f slurm-<jobname>-<jobid>.out        # everything the job prints
tail -f runs/dpo-<run>/train.log           # timestamped stage log
grep peak runs/dpo-<run>/train.log         # peak GPU memory per stage
```

## GPU usage of a running job

```bash
srun --jobid=<jobid> --overlap --pty nvidia-smi
srun --jobid=<jobid> --overlap --pty watch -n 5 nvidia-smi       # live, Ctrl+C to stop
srun --jobid=<jobid> --overlap --pty nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv -l 10

squeue -j <jobid> -o "%N"                  # the job's node
ssh <node>                                 # if allowed while your job runs there
```

During sampling, scoring and eval generation both GPUs should be busy. During DPO training only GPU 0 is used.

## Changing or stopping a job

```bash
scontrol update JobId=<jobid> MailUser=<email> MailType=END,FAIL   # add email alerts
scontrol show job <jobid> | grep -i mail                           # check them
scontrol update JobId=<jobid> TimeLimit=12:00:00   # lower the limit (raising needs an admin)
scancel <jobid>                            # cancel one job
scancel -u $USER                           # cancel all your jobs
scancel -u $USER -t PENDING                # cancel only the pending ones
```

Useful `MailType` values are `BEGIN`, `END`, `FAIL`, `TIME_LIMIT_80` and `ALL`.

## After a job

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
sacct -u $USER -S today --format=JobID,JobName,State,Elapsed     # today's jobs
seff <jobid>                               # CPU and memory efficiency, if installed
tail -n 50 runs/dpo-<run>/train.log
less runs/dpo-<run>/round_1/spotcheck.md   # read rewrites, not just metrics
```

| State | Meaning |
|---|---|
| `COMPLETED` | Finished normally |
| `FAILED` | Exited with an error; see the `.out` file |
| `TIMEOUT` | Hit the time limit; resubmit to resume |
| `OUT_OF_MEMORY` | Ran out of host RAM; raise `--mem` |
| `CANCELLED` | Stopped by `scancel` or an admin |

## Interactive GPU session

Use this for debugging. It needs a free GPU right away and ends when you disconnect.

```bash
sinteractive -p a100 --gres=gpu:1
source hpc_env.sh
python scripts/download_models.py --check  # do the models load offline on this node?
```

## TensorBoard

Run TensorBoard on the login node, not in the job:

```bash
source .venv/bin/activate
tensorboard --logdir runs --port 6006
# on your own machine:
ssh -L 6006:localhost:6006 <user>@<login-node>   # then open http://localhost:6006
```

## Gotchas

- **Disconnecting:** jobs keep running when you close SSH; anything you started on the login node doesn't, so use `tmux` there.
- **Wall time:** the partition limit is 2 days 12 h (`sinfo -p a100`), and shorter limits start sooner.
- **A100s per node:** each a100 node has 2 A100s, so a 2-GPU job needs a whole node's GPUs.
- **Run names:** use a new `--run` / `DEAI_RUN` when changing the base, reward, `n`, `train_limit` or NLI setting; `config.json` refuses mixed settings.
