"""
Rare Index - Configuration & Secrets Management
Handles secrets loading, runtime detection, and external access integration setup
"""

import streamlit as st
from typing import Optional, Dict, Any


# ============================================================================
# RUNTIME DETECTION
# ============================================================================

def running_in_snowflake() -> bool:
    """
    Detect if the app is running inside Snowflake's native Streamlit environment.
    
    Returns:
        bool: True if running in Snowflake, False otherwise
    """
    try:
        import _snowflake  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def configure_snowflake_external_access():
    """
    Configure external access integration for Snowflake runtime.
    Must be called before making external HTTP requests in Snowflake.
    
    Only has effect when running_in_snowflake() returns True.
    """
    if not running_in_snowflake():
        return
    
    try:
        import _snowflake
        _snowflake.configure(require_external_access_integrations=["RARE_EAI"])
    except Exception as e:
        st.warning(f"Failed to configure external access integration: {e}")


# ============================================================================
# SECRETS LOADER
# ============================================================================

def have_secret(path: str) -> bool:
    """
    Safely check if a secret exists without exposing its value.
    
    Args:
        path: Dot-notation path to secret (e.g., "newsapi.key" or "ebay.app_id")
        
    Returns:
        bool: True if secret exists and is not a placeholder, False otherwise
        
    Examples:
        >>> have_secret("newsapi.key")
        True
        >>> have_secret("nonexistent.secret")
        False
    """
    try:
        keys = path.split(".")
        value = st.secrets
        
        for key in keys:
            value = value.get(key)
            if value is None:
                return False
        
        # Check if it's a placeholder value
        if isinstance(value, str) and value.startswith("YOUR_"):
            return False
            
        return True
        
    except Exception:
        return False


def get_snowflake_config() -> Optional[Dict[str, str]]:
    """
    Load Snowflake connection configuration from secrets.
    
    Returns:
        Dict with connection parameters, or None if secrets are missing
    """
    required_keys = ["account", "user", "password", "warehouse", "database", "schema", "role"]
    
    if not all(have_secret(f"snowflake.{key}") for key in required_keys):
        return None
    
    return {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
        "role": st.secrets["snowflake"]["role"]
    }


def get_newsapi_key() -> Optional[str]:
    """
    Load NewsAPI key from secrets.
    
    Returns:
        API key string, or None if not configured
    """
    if not have_secret("newsapi.key"):
        return None
    return st.secrets["newsapi"]["key"]


def get_brickset_key() -> Optional[str]:
    """
    Load Brickset API key from secrets.
    
    Returns:
        API key string, or None if not configured
    """
    if not have_secret("brickset.key"):
        return None
    return st.secrets["brickset"]["key"]


def get_watchcharts_key() -> Optional[str]:
    """
    Load WatchCharts (The Watch API) key from secrets.
    
    Returns:
        API key string, or None if not configured
    """
    if not have_secret("watchcharts.key"):
        return None
    return st.secrets["watchcharts"]["key"]


def get_justcg_key() -> Optional[str]:
    """
    Load JusTCG API key from secrets.
    
    Returns:
        API key string, or None if not configured
    """
    if not have_secret("justcg.key"):
        return None
    return st.secrets["justcg"]["key"]


def get_pokeprice_key() -> Optional[str]:
    """
    Load Pokémon Price Tracker API key from secrets.
    
    Returns:
        API key string, or None if not configured
    """
    if not have_secret("pokeprice.key"):
        return None
    return st.secrets["pokeprice"]["key"]


def get_ebay_config() -> Optional[Dict[str, Any]]:
    """
    Load eBay API configuration from secrets.
    
    CURRENTLY DISABLED: eBay API not yet configured.
    
    Returns:
        Dict with eBay credentials and settings, or None if secrets are missing
    """
    # eBay integration coming soon - currently disabled
    return None
    
    # Uncomment when eBay credentials are available:
    # required_keys = ["app_id", "cert_id", "oauth_token"]
    # 
    # if not all(have_secret(f"ebay.{key}") for key in required_keys):
    #     return None
    # 
    # return {
    #     "app_id": st.secrets["ebay"]["app_id"],
    #     "cert_id": st.secrets["ebay"]["cert_id"],
    #     "dev_id": st.secrets["ebay"].get("dev_id", ""),
    #     "oauth_token": st.secrets["ebay"]["oauth_token"],
    #     "use_sandbox": st.secrets["ebay"].get("use_sandbox", True)
    # }


# ============================================================================
# SECRETS VALIDATION
# ============================================================================

def validate_secrets() -> Dict[str, bool]:
    """
    Check which secret groups are properly configured.
    
    Returns:
        Dict mapping secret group names to their availability status
        
    Example:
        >>> validate_secrets()
        {"snowflake": True, "newsapi": True, "brickset": True, "pokeprice": False}
    """
    return {
        "snowflake": get_snowflake_config() is not None,
        "newsapi": have_secret("newsapi.key"),
        "brickset": have_secret("brickset.key"),
        "pokeprice": have_secret("pokeprice.key"),
        "watchcharts": have_secret("watchcharts.key"),
        "justcg": have_secret("justcg.key"),
        # "ebay": get_ebay_config() is not None  # Coming soon
    }


def get_missing_secrets() -> list[str]:
    """
    Get a list of missing or incomplete secret groups.
    
    Returns:
        List of secret group names that are not properly configured
    """
    validation = validate_secrets()
    return [name for name, available in validation.items() if not available]
