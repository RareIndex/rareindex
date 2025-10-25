"""
Rare Index - Connectivity Test UI Component
Streamlit interface for testing external API connections
"""

import streamlit as st
from typing import Dict, Any
import time

from config import (
    running_in_snowflake,
    configure_snowflake_external_access,
    validate_secrets,
    get_newsapi_key,
    get_brickset_key,
    get_pokeprice_key,
    get_watchcharts_key,
    have_secret
)
from connectivity_tests import run_all_tests


# ============================================================================
# CACHED TEST RUNNER
# ============================================================================

@st.cache_data(ttl=120)
def run_cached_tests(force: int = 0) -> Dict[str, Any]:
    """
    Run connectivity tests with 2-minute caching.
    
    Args:
        force: Integer used to bust cache (increment to force re-run)
        
    Returns:
        Dict containing test results and metadata
    """
    # Configure external access if running in Snowflake
    if running_in_snowflake():
        configure_snowflake_external_access()
    
    # Get secrets
    newsapi_key = get_newsapi_key()
    brickset_key = get_brickset_key()
    pokeprice_key = get_pokeprice_key()
    watchcharts_key = get_watchcharts_key()
    
    # Run tests
    test_results = run_all_tests(
        newsapi_key=newsapi_key,
        brickset_key=brickset_key,
        pokeprice_key=pokeprice_key,
        watchcharts_key=watchcharts_key
    )
    
    # Add metadata
    return {
        "results": test_results,
        "timestamp": time.time(),
        "runtime": "snowflake" if running_in_snowflake() else "cloud/local"
    }


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_secret_status():
    """Render a table showing which secret groups are configured."""
    st.subheader("📋 Secrets Status")
    
    secrets_status = validate_secrets()
    
    # Core services
    st.markdown("**Core Services:**")
    col1, col2 = st.columns(2)
    
    with col1:
        status = "✅" if secrets_status["snowflake"] else "❌"
        st.metric("Snowflake", status)
        if not secrets_status["snowflake"]:
            st.caption("Missing connection info")
    
    with col2:
        status = "✅" if secrets_status["newsapi"] else "❌"
        st.metric("NewsAPI", status)
        if not secrets_status["newsapi"]:
            st.caption("Missing API key")
    
    # Collectibles APIs
    st.markdown("**Collectibles APIs:**")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        status = "✅" if secrets_status.get("brickset", False) else "❌"
        st.metric("Brickset (LEGO)", status)
        if not secrets_status.get("brickset", False):
            st.caption("Missing API key")
    
    with col4:
        status = "✅" if secrets_status.get("pokeprice", False) else "❌"
        st.metric("Pokémon TCG", status)
        if not secrets_status.get("pokeprice", False):
            st.caption("Missing API key")
    
    with col5:
        status = "✅" if secrets_status.get("watchcharts", False) else "❌"
        st.metric("Watch API", status)
        if not secrets_status.get("watchcharts", False):
            st.caption("Missing API key")
    
    # Runtime information
    runtime = "Snowflake Native Streamlit" if running_in_snowflake() else "Streamlit Cloud / Local"
    st.info(f"🖥️ **Runtime Environment:** {runtime}")
    
    if running_in_snowflake():
        st.caption("External Access Integration (RARE_EAI) will be configured automatically")


def render_test_results(test_data: Dict[str, Any]):
    """
    Render the results of connectivity tests.
    
    Args:
        test_data: Dict containing test results and metadata
    """
    st.subheader("🔌 Connectivity Test Results")
    
    results = test_data["results"]
    timestamp = test_data["timestamp"]
    
    # Show when tests were run
    st.caption(f"Last run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
    
    # Helper function to render a test result
    def render_result(api_name: str, display_name: str):
        api_result = results.get(api_name, {})
        success = api_result.get("success")
        message = api_result.get("message", "unknown")
        
        st.markdown(f"### {display_name}")
        if success is None:
            st.warning(f"⏭️ **Skipped:** {message}")
        elif success:
            st.success(f"✅ **Connected:** {message}")
        else:
            st.error(f"❌ **Failed:** {message}")
    
    # NewsAPI test results
    render_result("newsapi", "NewsAPI")
    
    # Brickset test results
    render_result("brickset", "Brickset API (LEGO)")
    
    # Pokémon Price Tracker test results
    render_result("pokeprice", "Pokémon TCG Price Tracker")
    
    # The Watch API test results
    render_result("watchcharts", "The Watch API")
    
    # eBay (disabled)
    st.markdown("### eBay API")
    st.info("⏭️ **Not configured yet:** eBay integration coming soon")


def render_connectivity_tests():
    """
    Main function to render the complete connectivity test interface.
    
    Can be called from sidebar or main page.
    """
    st.title("🔧 Connectivity Tests")
    
    st.markdown("""
    This page tests external API connectivity to ensure all integrations are working properly.
    Tests are cached for 2 minutes to avoid excessive API calls.
    """)
    
    # Render secrets status
    render_secret_status()
    
    st.divider()
    
    # Initialize force counter in session state
    if "test_force_counter" not in st.session_state:
        st.session_state.test_force_counter = 0
    
    # Buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        run_tests = st.button("▶️ Run Tests", use_container_width=True, type="primary")
    
    with col2:
        rerun_tests = st.button("🔄 Re-run Tests (Bypass Cache)", use_container_width=True)
    
    # Handle button clicks
    if rerun_tests:
        st.session_state.test_force_counter += 1
        run_tests = True  # Also trigger test run
    
    if run_tests:
        with st.spinner("Running connectivity tests..."):
            try:
                test_data = run_cached_tests(force=st.session_state.test_force_counter)
                st.divider()
                render_test_results(test_data)
            except Exception as e:
                st.error(f"❌ Test execution failed: {str(e)}")
                st.exception(e)
    else:
        st.info("👆 Click 'Run Tests' to check API connectivity")


# ============================================================================
# SIDEBAR WIDGET (COMPACT VERSION)
# ============================================================================

def render_connectivity_sidebar():
    """
    Compact version for sidebar display.
    Shows quick status and link to full test page.
    """
    with st.sidebar.expander("🔌 API Connectivity", expanded=False):
        secrets_status = validate_secrets()
        
        # Quick status indicators
        st.markdown("**Quick Status:**")
        for name, available in secrets_status.items():
            icon = "✅" if available else "❌"
            st.caption(f"{icon} {name.title()}")
        
        # Link to full test page
        st.markdown("---")
        if st.button("Run Full Tests", use_container_width=True):
            st.switch_page("pages/connectivity.py")  # Adjust path as needed
