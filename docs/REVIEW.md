# Independent review — findings and resolutions

An independent reviewer (fresh context, report-only) audited the package for math/convention bugs,
security, silent failures and edge cases. It confirmed: URDF `rpy` convention, MuJoCo `wxyz` quaternions,
identical inertial frames in URDF/MJCF, correct revolute pivot/angle math, continuous joints without limits,
`safe_name` + `quoteattr` on every XML attribute, viewer JSON escaping + `textContent`, `allow_pickle=False`
everywhere, and MCP path confinement.

| Severity | Finding | Resolution | Regression test |
|---|---|---|---|
| HIGH | Tracks invisible in frame 0 got rest position (0,0,0), corrupting poses, boxes and mass | Pipeline drops tracks without a rest observation and records a note | `test_tracks_hidden_in_first_frame_do_not_corrupt_the_model` |
| MEDIUM | URDF root frame offset from MJCF root by the root's box centre | Root link frame is the capture world origin in both formats | `test_urdf_and_mjcf_agree_on_world_poses` (full FK comparison) |
| MEDIUM | `content: null` LLM replies and raw `http.client` errors crashed instead of falling back | Content type checked; transport errors retried; any annotation failure falls back with a note | `test_llm_non_text_content_falls_back`, `test_llm_unexpected_exception_still_falls_back` |
| MEDIUM | 1–2 track user labels became static fake parts that could win the root | Labels with < 6 tracks are ignored with a note | `test_tiny_user_labels_are_ignored_not_made_root` |
| LOW | HTTP redirects would forward the API key and bypass the https check | Redirects refused | `test_redirects_are_refused` |
| LOW | Part named `world` broke the MJCF | Reserved names are prefixed | `test_world_is_a_reserved_name` |
| LOW | < 6 frames displayed "±0.00°" (false certainty) | Uncertainty is `null`/n-a with a warning | `test_too_few_frames_reports_unknown_uncertainty_not_zero` |
| LOW | `--stride 0` crashed | Argparse + config validation | `test_cli_rejects_zero_stride` |
| LOW | CDN scripts without SRI; LLM notes unescaped in Markdown | SRI `sha384` + `crossorigin`; notes escaped | `test_markdown_escaping_of_untrusted_notes` |

Verdict after fixes: no open CRITICAL/HIGH/MEDIUM findings; 80 tests pass.
