## Twin accuracy vs humanoid effort

| axis error | feasible tasks | succeed | grasp missed | median extra grasp force | worst extra grasp force | median extra hand deviation |
|---|---|---|---|---|---|---|
| measured twin | 10 | 100% | 0 | +0 N | +6 N | +0 mm |
| twin + 2° | 10 | 100% | 0 | -0 N | +2 N | -1 mm |
| twin + 5° | 10 | 100% | 0 | -0 N | +5 N | +3 mm |
| twin + 10° | 10 | 80% | 2 | -0 N | +8 N | +21 mm |
| twin + 15° | 10 | 60% | 2 | +2 N | +25 N | +41 mm |

| axis error | drawers and slides: extra force | doors and lids: extra force |
|---|---|---|
| measured twin | +0 N (max -0) | +2 N (max +6) |
| twin + 2° | -0 N (max -0) | +1 N (max +2) |
| twin + 5° | -0 N (max -0) | -0 N (max +5) |
| twin + 10° | -0 N (max -0) | +4 N (max +8) |
| twin + 15° | -2 N (max +2) | +18 N (max +25) |

## +1° (run separately, raw rows in `task_bench_apollo_1deg.json`)

| axis error | feasible tasks | succeed | grasp missed | median extra grasp force | worst extra grasp force | median extra hand deviation |
|---|---|---|---|---|---|---|
| twin + 1° | 10 | 100% | 0 | +0 N | +5 N | +0 mm |

