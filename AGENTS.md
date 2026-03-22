# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a clean starting point with no application code checked in yet. Keep the root minimal and introduce a predictable layout as the project grows:

- `src/` for application or library code
- `tests/` for automated tests
- `docs/` for design notes and architecture decisions
- `assets/` for static files such as images or sample data

Prefer grouping code by feature or domain instead of creating deep utility folders too early.

## Build, Test, and Development Commands
There is no build system configured yet. When tooling is added, expose it through documented commands and keep them stable. Recommended baseline:

- `make setup` or `npm install` to install dependencies
- `make test` or `npm test` to run the full test suite
- `make lint` or `npm run lint` to enforce style checks
- `make dev` or `npm run dev` to start local development

Add new commands to this file and to the project README when they become available.

## Coding Style & Naming Conventions
Use consistent formatting from the start:

- Indentation: 2 spaces for Markdown, JSON, YAML, and frontend code; 4 spaces for Python
- Filenames: lowercase with hyphens for docs (`architecture-overview.md`), snake_case for Python modules, camelCase only when required by the language ecosystem
- Keep modules focused and avoid large multi-purpose files

Adopt an automatic formatter and linter early, such as `prettier` and `eslint` for JavaScript/TypeScript or `black` and `ruff` for Python.

## Testing Guidelines
Place tests under `tests/` and mirror the source layout where practical. Name tests after the unit under test, for example `tests/test_parser.py` or `tests/user-service.test.ts`.

Every new feature should include tests for expected behavior and key failure paths. If coverage tooling is added, document the target threshold here.

## Commit & Pull Request Guidelines
There is no Git history yet, so use a simple convention now:

- Commit messages: imperative, concise, and scoped, for example `Add API client skeleton`
- Keep unrelated changes out of the same commit
- Pull requests should include a short summary, testing notes, and linked issues when applicable

Include screenshots or sample output for UI or CLI behavior changes.

## Security & Configuration Tips
Do not commit secrets, local environment files, or generated credentials. Keep runtime configuration in environment variables and provide a checked-in example file such as `.env.example` when configuration is introduced.
