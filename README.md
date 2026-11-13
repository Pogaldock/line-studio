# line-studio

Photo -> single-continuous-line drawing options -> captioned 30 s reveal video. Driven from the phone page
symbiozefpv.com/admin/lines (admin only) through the site Worker; all compute runs here on GitHub Actions.

- `engine/` the pipeline (hydrant technique + route order + light humanize, the locked look). `engine/options.json` = the options shown per photo.
- `engine/studio.py` orchestrator: `trace` (all options, stills + progress to the job release) and `render` (chosen option -> video).
- `.github/workflows/trace.yml`, `render.yml` started by the Worker (workflow_dispatch). Each job = one release `job-<id>`.
- `tools/ref/` reference drawings; `tools/regress.py` checks the runner reproduces them exactly.
- Models: the line model is in `engine/models/`; the others download from their official releases (`engine/models/models.txt`, sha256 checked) and are cached on the runner.
