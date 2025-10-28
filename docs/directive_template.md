Directive: Rare Index – Feature <X>

Goal:
- <One sentence: What the feature does>
- User impact: <One sentence>

Constraints:
- Python 3.11, Streamlit, Snowpark
- Code in /app/ for UI, /etl/ for jobs, /tests/ for tests
- No secrets in code; use os.getenv()
- Top-level imports only
- Snowflake-compatible: No relative imports
- Typed functions, 90% test coverage
- Log external calls to audit table

Deliverables:
- File paths
- Code blocks
- Run instructions
- Rollback
- Assumptions/Risks

Acceptance:
- CI passes
- No local/Snowflake errors
- PR <200 lines