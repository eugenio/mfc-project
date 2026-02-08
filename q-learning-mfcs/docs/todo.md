# TODO

## Pipeline Failure — MR !21 (`feature/cad-stack-design`)

- [ ] Investigate pipeline #220 failure (SHA `fd2bbf1`, duration 47s)
  - Pipeline URL: https://gitlab-runner.tail301d0a.ts.net/uge/mfc-project/-/pipelines/220
  - MR URL: https://gitlab-runner.tail301d0a.ts.net/uge/mfc-project/-/merge_requests/21
  - Triggered by: `merge_request_event` on `refs/merge-requests/21/head`
  - Check job logs for root cause (likely missing CadQuery dependency in CI runner)
  - Fix CI config or mark CadQuery-dependent tests to skip in CI environment
