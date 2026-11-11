## Twin accuracy vs humanoid effort

| axis error | feasible tasks | succeed | grasp missed | median extra grasp force | worst extra grasp force | median extra hand deviation |
|---|---|---|---|---|---|---|
| measured twin | 7 | 100% | 0 | +0 N | +2 N | +0 mm |
| twin + 2° | 7 | 0% | 0 | +19 N | +24 N | -0 mm |
| twin + 5° | 7 | 0% | 0 | +50 N | +77 N | +7 mm |
| twin + 10° | 7 | 0% | 5 | +79 N | +120 N | +8 mm |
| twin + 15° | 7 | 0% | 5 | +125 N | +181 N | +19 mm |

Infeasible for this robot even with the true joint (excluded): dresser/door, fridge/left_door, fridge/right_door.

| axis error | drawers and slides: extra force | doors and lids: extra force |
|---|---|---|
| measured twin | +0 N (max +2) | -2 N (max +0) |
| twin + 2° | +19 N (max +20) | +17 N (max +24) |
| twin + 5° | +50 N (max +50) | +44 N (max +77) |
| twin + 10° | n/a | +79 N (max +120) |
| twin + 15° | n/a | +125 N (max +181) |

## Fine sweep near the threshold (generic arm)

Raw rows: `task_bench_fine.json`.

| axis error | feasible tasks | succeed | grasp missed | median extra grasp force | worst extra grasp force | median extra hand deviation |
|---|---|---|---|---|---|---|
| measured twin | 7 | 100% | 0 | +0 N | +2 N | +0 mm |
| twin + 0.5° | 7 | 100% | 0 | +4 N | +6 N | -0 mm |
| twin + 1° | 7 | 57% | 0 | +9 N | +11 N | -1 mm |
| twin + 1.5° | 7 | 14% | 0 | +14 N | +15 N | -1 mm |

Infeasible for this robot even with the true joint (excluded): dresser/door, fridge/left_door, fridge/right_door.

