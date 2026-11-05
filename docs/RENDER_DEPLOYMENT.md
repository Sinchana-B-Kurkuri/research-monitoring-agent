# Update the existing Render deployment

The dashboard code is on `main`. The reports are repository documentation and are not served by the application. Publishing Git commits does not copy your local PostgreSQL data, pilot profiles, or `.env` to Render.

## Check the existing service

1. Open the existing **Web Service** in Render. In **Settings**, confirm the connected GitHub repository is `Sinchana-B-Kurkuri/research-monitoring-agent`, the branch is `main`, and commands run from the repository root. Reuse the service rather than creating a new one.
2. In **Environment**, keep its existing `DATABASE_URL` for the hosted PostgreSQL database. For AI summaries, `GEMINI_API_KEY` must be set on Render. Check any `LLM_MODEL` override: it takes precedence over the repository default of `gemini-3.5-flash-lite`. Keep credentials in Render, outside Git.
3. The deployment must install the package **in editable mode from the checkout**: the onboarding and monitoring workers locate scripts by repository-relative paths. If the existing build command does not do this, use:

   ```bash
   python -m pip install -e .
   ```

4. The updated dashboard uses a `background_jobs` table. `scripts/run_ui.py` creates that table if missing, but prints a private operator link and is intended for local use. If the hosted service starts Uvicorn directly and has not already provisioned this table, use the following **Start Command**. It performs the same table check without printing a private link:

   ```bash
   python -c "from research_monitor.database.models import BackgroundJob; from research_monitor.database.session import engine; BackgroundJob.__table__.create(engine, checkfirst=True)" && python -m uvicorn research_monitor.ui.app:app --host 0.0.0.0 --port "$PORT" --workers 1
   ```

   The hosted database must already have the application's tables, pgvector, and the `training_labels` view. If startup reports missing tables or insufficient database privileges, inspect the hosted schema and back up the database before applying any migration. Do not replace the database to make the new UI load.

## Publish and verify

From the project directory in PowerShell:

```powershell
git status --short --branch
git push origin main
git rev-parse --short HEAD
```

This publishes the committed code and reports. Review the documents under `docs/reports/` before pushing if the GitHub repository is public.

In Render, open **Deploys** and check that the deployed commit matches `main`. If the push did not trigger a deployment, choose **Manual Deploy → Deploy latest commit**. Wait for **Live**, then refresh the hosted URL and check `/health`, the **Workspace** or **Initialize a pilot** page, and a hosted pilot's dashboard. See [Render’s deploy documentation](https://render.com/docs/deploys#automatic-deploys).

If the page loads but monitoring or onboarding fails, check the service logs and the pilot's job history. A successful `/health` only proves database connectivity. Onboarding also needs the reference-paper embeddings available to the hosted database or filesystem; monitoring needs enough memory for the embedding worker and network access to arXiv.

For browser actions on the default Render domain, the application uses Render's automatically supplied `RENDER_EXTERNAL_URL` to recognize the public HTTPS origin if the proxy presents the request as HTTP. If your service uses a custom domain and the app sees the connection as HTTP, set `PUBLIC_ORIGIN` to the public HTTPS origin (scheme, hostname and optional port only).

Existing hosted personal links use the hosted database. The operator key and generated evaluation files are filesystem state. Without a persistent disk or a persistent `WORKSPACE_KEY_PATH`, a redeploy can replace the operator key even when all pilot records remain in PostgreSQL. See [Render persistent disks](https://render.com/docs/disks).
