# Merge conflict lab

This folder is a small, realistic Git exercise for the merge-conflict lesson.
The shared benchmark is intentionally simple: it measures the bundled quick-sort
demo at three input sizes.

Two developers then make valid but competing changes:

- `lesson/fast-local-feedback` shortens the timeout, uses fewer repeats, and
  increases parallelism so local experiments finish quickly.
- `lesson/stable-ci-measurements` allows more time, uses more repeats, and keeps
  parallelism low so CI results are easier to compare.

## Reproduce the conflict

Start from the fast-feedback branch and merge the stable-CI branch:

```bash
git switch lesson/fast-local-feedback
git merge lesson/stable-ci-measurements
```

Git will stop with conflicts in both `benchmark_plan.yaml` and this README.
Resolve each file by choosing the policy that fits the environment, then run:

```bash
git add examples/merge_conflicts
git commit
python -m tembench validate --config examples/merge_conflicts/benchmark_plan.yaml
```

The branches are deliberately left unmerged so the conflict can be created on
camera and resolved step by step.

## Shared benchmark

```bash
python -m tembench validate --config examples/merge_conflicts/benchmark_plan.yaml
```

The configuration uses wall-clock timing because the bundled demo prints a
number but does not emit TempoBench's optional `TEMPOBENCH_MS` marker.

## Branch policy notes

CI policy: favor stable comparisons with a 60-second timeout, 7 repeats, and 2
workers.
