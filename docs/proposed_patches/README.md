# Proposed patches (not applied to any branch, not committed)

Reference copies of the missing `traj_geom/data/loaders.py` module, kept
here so they aren't lost even though they were built/tested in disposable
`git worktree`s that get torn down at the end of a session.

- **`main_branch_data_loaders.py`** — already applied at
  `src/traj_geom/data/loaders.py` in this checkout (main branch). High
  confidence: it's a verbatim port of `notebooks/01_mvp.ipynb` cells 10-11,
  and running `python -m scripts.run_pararule` against it reproduces the
  README's exact published numbers (winding~depth rho=+0.20, steps~depth
  rho=+0.80, N=4).

- **`two_scale_branch_data_loaders.py`** — a best-effort reconstruction for
  `origin/two-scale-real-fix` (the open PR branch), covering
  `load_pararule_real` and `pararule_depth_of`. Lower confidence than the
  main-branch version: there's no surviving notebook cell for this one, so
  it's built from (1) `tests/test_two_scale.py`'s exact contract for
  `pararule_depth_of` — verified, that test passes with this file in place —
  and (2) the real `qbao775/PARARULE-Plus-Depth-{2..5}` HF dataset schema
  (confirmed via the dataset viewer to have `id`/`context`/`question`/`label`/
  `meta.QDep` fields, matching the function's assumptions). What's NOT
  verified: whether the original `load_pararule_real` actually drew from
  these same per-depth-split repos or a different combined one. Full 12-test
  PR-branch suite (`11 passed, 1 skipped [ripser]`) passes with this file
  dropped into `src/traj_geom/data/loaders.py` on that branch, in a
  disposable worktree, 2026-07-17.

To apply the two-scale one: `git worktree add <path> origin/two-scale-real-fix`,
copy this file to `<path>/src/traj_geom/data/loaders.py` (plus an empty
`__init__.py` in that dir), `uv sync --extra dev`, `pytest -q` to confirm,
then it's a normal uncommitted change on that branch — commit/push is
whoever owns that branch's call, not made here.
