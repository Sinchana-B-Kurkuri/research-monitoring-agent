# Architectural Decisions

## Phase 0 Foundation

- **Language**: Python
- **Layout**: Modular `src/` layout to separate code from configuration and tests.
- **Configuration**: Environment variables are used for settings and secrets (loaded from `.env` via `python-dotenv`). No secrets are committed to the repository.
- **Database**: PostgreSQL/pgvector will be introduced in a later phase.
- **Testing**: Tests are required for important logic. `pytest` is the chosen framework.
- **Version Control**: Git is managed manually by the developer.
