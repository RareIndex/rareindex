# Rare Index - Connectivity Tests Integration

## Overview
This module provides secure secrets management and external API connectivity testing for the Rare Index platform.

## Files Included

```
project/
├── config.py                    # Secrets loader & runtime detection
├── connectivity_tests.py        # HTTP smoke tests for APIs
├── ui_connectivity.py          # Streamlit UI components
├── pages/
│   └── connectivity.py         # Standalone test page
├── app_example.py              # Example main app integration
├── test_connectivity.py        # CLI test script
├── requirements.txt            # Python dependencies
└── .gitignore                  # Secrets protection
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Secrets

Create `.streamlit/secrets.toml` in your project root:

```toml
[snowflake]
account = "your_account.region"
user = "RAREIDX_SVC"
password = "your_password"
warehouse = "RARE_WH"
database = "RAREINDEX_DB"
schema = "PUBLIC"
role = "RARE_APP_ROLE"

[newsapi]
key = "your_newsapi_key"

[ebay]
app_id = "your_app_id"
cert_id = "your_cert_id"
dev_id = "your_dev_id"
oauth_token = "your_oauth_token"
use_sandbox = true
```

**CRITICAL:** Add `.streamlit/secrets.toml` to `.gitignore` immediately!

### 3. Deploy to Streamlit Cloud

1. Go to your app settings
2. Navigate to **Secrets** section
3. Paste the contents of your `secrets.toml` file
4. Click **Save**

### 4. Snowflake External Access (If Deploying to Snowflake)

Ensure you've run the Snowflake setup script that creates:
- Network rule: `RARE_EGRESS_RULE`
- External access integration: `RARE_EAI`
- Grants to `RARE_APP_ROLE`

## Usage

### In Your Main App

```python
from ui_connectivity import render_connectivity_sidebar
from config import validate_secrets, configure_snowflake_external_access

# Add to sidebar
render_connectivity_sidebar()

# Check secrets before using APIs
secrets_status = validate_secrets()
if not secrets_status["newsapi"]:
    st.warning("NewsAPI not configured")

# Configure external access (Snowflake only)
configure_snowflake_external_access()
```

### Standalone Test Page

Create `pages/connectivity.py` to add a dedicated test page to your multipage app.

### CLI Testing (Local Development)

```bash
python test_connectivity.py
```

## API Reference

### `config.py`

**Functions:**
- `running_in_snowflake() -> bool` - Detect Snowflake runtime
- `configure_snowflake_external_access()` - Set up RARE_EAI integration
- `have_secret(path: str) -> bool` - Check if secret exists (safe)
- `get_snowflake_config() -> dict | None` - Load Snowflake credentials
- `get_newsapi_key() -> str | None` - Load NewsAPI key
- `get_ebay_config() -> dict | None` - Load eBay credentials
- `validate_secrets() -> dict` - Check all secret groups
- `get_missing_secrets() -> list` - Get list of missing secrets

### `connectivity_tests.py`

**Functions:**
- `test_newsapi(key: str) -> tuple[bool, str]` - Test NewsAPI connection
- `test_ebay(oauth_token: str, sandbox: bool) -> tuple[bool, str]` - Test eBay API
- `run_all_tests(...) -> dict` - Run all available tests

### `ui_connectivity.py`

**Functions:**
- `render_connectivity_tests()` - Full-page test interface
- `render_connectivity_sidebar()` - Compact sidebar widget
- `run_cached_tests(force: int) -> dict` - Cached test runner (2min TTL)

## Security Notes

✅ **Never print secrets** - All functions use safe checks  
✅ **Gitignore configured** - Secrets files excluded from version control  
✅ **No secrets in logs** - Error messages don't expose tokens  
✅ **Timeout protection** - All HTTP requests have 15s timeout  

## Troubleshooting

### "Role 'RARE_APP_ROLE' does not exist"
Run the Snowflake reconcile script to create the role and grant it to your user.

### "External access integration not found"
Ensure `RARE_EAI` exists in Snowflake and is granted to your role.

### API Test Failures
- Check secrets are correctly configured
- Verify API keys are valid and not expired
- For eBay: OAuth tokens expire regularly
- Check network connectivity / firewall rules

### Timeout Errors
- Increase timeout in `connectivity_tests.py` (default: 15s)
- Check if external domains are blocked by firewall
- In Snowflake: Verify network rule includes the domain

## Next Steps

1. ✅ Run reconcile SQL script in Snowflake
2. ✅ Configure secrets locally or in Streamlit Cloud
3. ✅ Test connectivity using the UI or CLI script
4. ✅ Integrate into your main app
5. ✅ Add error handling for missing secrets in features

## Support

For issues or questions:
1. Check the troubleshooting guide above
2. Review Snowflake grants: `SHOW GRANTS TO ROLE RARE_APP_ROLE;`
3. Verify secrets: Visit the Connectivity Tests page
