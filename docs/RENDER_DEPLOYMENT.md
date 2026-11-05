# Update the existing Render deployment

The dashboard code is on `main`. The reports are repository documentation and are not served by the application. Publishing Git commits does not copy your local PostgreSQL data, pilot profiles, or `.env` to Render.

## Check the existing service

1. Open the existing **Web Service** in Render. In **Settings**, confirm the connected GitHub repository is `Sinchana-B-Kurkuri/research-monitoring-agent`, the branch is `main`, and commands run from the repository root. Reuse the service rather than creating a new one.
2. In **Environment**, keep its existing `DATABASE_URL` for the hosted PostgreSQL database. For AI summaries, `GEMINI_API_KEY` must be set on Render. Check any `LLM_MODEL` override: it takes precedence over the repository default of `gemini-3.5-flash-lite`. Keep credentials in Render, outside Git.
3. Set the **Build Command** to install CPU PyTorch and download the configured embedding model at build time. Use the checkout in editable mode so the workers can find the repository scripts. The build and start commands use the same Hugging Face cache:

   ```bash
   python -m pip install --index-url https://download.pytorch.org/whl/cpu torch && python -m pip install -e . && HF_HOME="$PWD/.hf-cache" python -c "from research_monitor.config import config; from sentence_transformers import SentenceTransformer; SentenceTransformer(config.EMBEDDING_MODEL, device='cpu')"
   ```

4. Set the **Start Command** and set **Health Check Path** to `/health`. Keep one Uvicorn worker and limit the CPU thread pools. The startup check creates the `background_jobs` table if needed. The local launcher prints a private operator link, so use this command for the web service:

   ```bash
   python -c "from research_monitor.database.models import BackgroundJob; from research_monitor.database.session import engine; BackgroundJob.__table__.create(engine, checkfirst=True)" && HF_HOME="$PWD/.hf-cache" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m uvicorn research_monitor.ui.app:app --host 0.0.0.0 --port "$PORT" --workers 1
   ```

   The hosted database must already have the application's tables, pgvector, and the `training_labels` view. If startup reports missing tables or insufficient database privileges, inspect the hosted schema and back up the database before applying any migration.

5. Verify the reference corpus in the hosted database using its SQL console. The default deployment requires 5,000 reference embeddings. Git contains the code and an empty data directory, so a new hosted database does not automatically receive the local corpus.

   ```sql
   SELECT count(*) AS reference_embeddings
   FROM papers
   WHERE is_reference AND embedding IS NOT NULL;
   ```

   If the sample is missing, run `scripts/run_phase3_loader.py` from the local checkout with its existing `data/corpus.parquet` and `data/embeddings.npy` and with `DATABASE_URL` explicitly set to the intended hosted database for that shell. Back up the hosted database first. The loader inserts missing paper IDs and preserves existing papers and pilot records. If existing papers are present but marked incorrectly, the loader will preserve those flags. Compare the database with the original reference sample before correcting flags.

6. Check the service RAM. A Linux test with the real model and 5,000 PostgreSQL reference embeddings hit an out-of-memory kill at 512 MiB even with CPU PyTorch and a warm cache. The same form completed with a 2 GiB limit. Render lists both Free and Starter at 512 MB and Standard (`1c-2g`) at 2 GB. Use the 2 GB tier or more to keep the current embedding workflow. See [Render compute plans](https://render.com/docs/compute-plans#web-service-plans).

## Publish and verify

From the project directory in PowerShell:

```powershell
git status --short --branch
git push origin main
git rev-parse --short HEAD
```

This publishes the committed code and reports. Review the documents under `docs/reports/` before pushing if the GitHub repository is public.

In Render, open **Deploys** and check that the deployed commit matches `main`. If the push did not trigger a deployment, choose **Manual Deploy → Deploy latest commit**. Wait for **Live**, then refresh the hosted URL and check `/health`, the **Workspace** or **Initialize a pilot** page, and a hosted pilot's dashboard. See [Render’s deploy documentation](https://render.com/docs/deploys#automatic-deploys).

After changing the build command, use **Manual Deploy → Clear build cache & deploy** once so the CPU wheel and model cache are rebuilt. If the page loads but onboarding or monitoring fails, check the service logs and Events at the time of the request. A successful `/health` confirms database connectivity. A `502` followed by a new Uvicorn process means the application was unavailable at the gateway. Look for `Out of memory`, `SIGKILL`, exit code `137`, or failing health checks to identify why. The build-time cache avoids a first-request model download, but the model still needs memory during inference. Monitoring also requires network access to arXiv. See [Render troubleshooting](https://render.com/docs/troubleshooting-deploys#502-bad-gateway).

For browser actions on the default Render domain, the application uses Render's automatically supplied `RENDER_EXTERNAL_URL` to recognize the public HTTPS origin if the proxy presents the request as HTTP. If your service uses a custom domain and the app sees the connection as HTTP, set `PUBLIC_ORIGIN` to the public HTTPS origin (scheme, hostname and optional port only).

Existing hosted personal links use the hosted database. The operator key and generated evaluation files are filesystem state. Without a persistent disk or a persistent `WORKSPACE_KEY_PATH`, a redeploy can replace the operator key even when all pilot records remain in PostgreSQL. See [Render persistent disks](https://render.com/docs/disks).
