"""
Rare Index - External API Connectivity Tests
Smoke tests for NewsAPI and eBay API connectivity
"""

import requests
from typing import Tuple


# ============================================================================
# HTTP SMOKE TESTS
# ============================================================================

def test_newsapi(key: str) -> Tuple[bool, str]:
    """
    Test connectivity to NewsAPI.
    
    Args:
        key: NewsAPI API key
        
    Returns:
        Tuple of (success: bool, message: str)
        - (True, "ok") if API responds successfully
        - (False, "error message") if request fails
        
    Example:
        >>> test_newsapi("valid_api_key")
        (True, "ok")
        >>> test_newsapi("invalid_key")
        (False, "status 401: Unauthorized")
    """
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "country": "us",
            "pageSize": 1,
            "apiKey": key
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            return (True, "ok")
        else:
            # Extract error message from response if available
            try:
                error_data = response.json()
                error_msg = error_data.get("message", response.text[:100])
            except Exception:
                error_msg = response.text[:100]
            
            return (False, f"status {response.status_code}: {error_msg}")
            
    except requests.exceptions.Timeout:
        return (False, "request timeout (>15s)")
    except requests.exceptions.ConnectionError:
        return (False, "connection failed - check network/firewall")
    except Exception as e:
        return (False, f"error: {str(e)[:100]}")


def test_brickset(api_key: str) -> Tuple[bool, str]:
    """
    Test connectivity to Brickset API.
    
    Args:
        api_key: Brickset API key
        
    Returns:
        Tuple of (success: bool, message: str)
        - (True, "ok") if API responds successfully
        - (False, "error message") if request fails
        
    Example:
        >>> test_brickset("valid_api_key")
        (True, "ok")
        >>> test_brickset("invalid_key")
        (False, "status 401: Invalid API key")
    """
    try:
        # Test with getSets endpoint (minimal data request)
        url = "https://brickset.com/api/v3.asmx/getSets"
        params = {
            "apiKey": api_key,
            "userHash": "",  # Not required for basic getSets
            "params": "{}"   # Empty params to get minimal response
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            # Check if response is valid JSON with status
            try:
                data = response.json()
                if data.get("status") == "success":
                    return (True, "ok")
                else:
                    error_msg = data.get("message", "Unknown error")
                    return (False, f"API error: {error_msg}")
            except Exception:
                return (True, "ok")  # Got 200, assume success even if JSON parse fails
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("message", response.text[:100])
            except Exception:
                error_msg = response.text[:100]
            
            return (False, f"status {response.status_code}: {error_msg}")
            
    except requests.exceptions.Timeout:
        return (False, "request timeout (>15s)")
    except requests.exceptions.ConnectionError:
        return (False, "connection failed - check network/firewall")
    except Exception as e:
        return (False, f"error: {str(e)[:100]}")


def test_pokeprice(api_key: str) -> Tuple[bool, str]:
    """
    Test connectivity to Pokémon Price Tracker API.
    
    Args:
        api_key: Pokémon Price Tracker API key
        
    Returns:
        Tuple of (success: bool, message: str)
        - (True, "ok") if API responds successfully
        - (False, "error message") if request fails
        
    Example:
        >>> test_pokeprice("valid_api_key")
        (True, "ok")
        >>> test_pokeprice("invalid_key")
        (False, "status 401: Unauthorized")
    """
    try:
        # Test with a simple search endpoint
        url = "https://api.pokemontcg.io/v2/cards"
        headers = {
            "X-Api-Key": api_key
        }
        params = {
            "page": 1,
            "pageSize": 1  # Minimal data request
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            return (True, "ok")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("message", response.text[:100])
            except Exception:
                error_msg = response.text[:100]
            
            return (False, f"status {response.status_code}: {error_msg}")
            
    except requests.exceptions.Timeout:
        return (False, "request timeout (>15s)")
    except requests.exceptions.ConnectionError:
        return (False, "connection failed - check network/firewall")
    except Exception as e:
        return (False, f"error: {str(e)[:100]}")


def test_watchcharts(api_key: str) -> Tuple[bool, str]:
    """
    Test connectivity to The Watch API (WatchCharts).
    
    Args:
        api_key: The Watch API key
        
    Returns:
        Tuple of (success: bool, message: str)
        - (True, "ok") if API responds successfully
        - (False, "error message") if request fails
    """
    try:
        # Test with brands endpoint (minimal request)
        url = "https://api.thewatchapi.com/v1/brands"
        headers = {
            "x-api-key": api_key
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return (True, "ok")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("message", response.text[:100])
            except Exception:
                error_msg = response.text[:100]
            
            return (False, f"status {response.status_code}: {error_msg}")
            
    except requests.exceptions.Timeout:
        return (False, "request timeout (>15s)")
    except requests.exceptions.ConnectionError:
        return (False, "connection failed - check network/firewall")
    except Exception as e:
        return (False, f"error: {str(e)[:100]}")


def test_ebay(oauth_token: str, sandbox: bool = True) -> Tuple[bool, str]:
    """
    Test connectivity to eBay API.
    
    CURRENTLY DISABLED: eBay API not yet configured.
    
    Args:
        oauth_token: eBay OAuth token
        sandbox: If True, use sandbox API; if False, use production API
        
    Returns:
        Tuple of (False, "not configured")
    """
    return (False, "eBay API not yet configured - coming soon")
    
    # Uncomment when eBay credentials are available:
    # try:
    #     base_url = (
    #         "https://api.sandbox.ebay.com" if sandbox
    #         else "https://api.ebay.com"
    #     )
    #     
    #     url = f"{base_url}/buy/browse/v1/item_summary/search"
    #     headers = {
    #         "Authorization": f"Bearer {oauth_token}",
    #         "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"
    #     }
    #     params = {
    #         "q": "lego",
    #         "limit": 1
    #     }
    #     
    #     response = requests.get(url, headers=headers, params=params, timeout=15)
    #     
    #     if response.status_code == 200:
    #         return (True, "ok")
    #     else:
    #         try:
    #             error_data = response.json()
    #             errors = error_data.get("errors", [])
    #             if errors:
    #                 error_msg = errors[0].get("message", response.text[:100])
    #             else:
    #                 error_msg = error_data.get("message", response.text[:100])
    #         except Exception:
    #             error_msg = response.text[:100]
    #         
    #         return (False, f"status {response.status_code}: {error_msg}")
    #         
    # except requests.exceptions.Timeout:
    #     return (False, "request timeout (>15s)")
    # except requests.exceptions.ConnectionError:
    #     return (False, "connection failed - check network/firewall")
    # except Exception as e:
    #     return (False, f"error: {str(e)[:100]}")


# ============================================================================
# COMBINED TEST RUNNER
# ============================================================================

def run_all_tests(newsapi_key: str = None, 
                  brickset_key: str = None,
                  pokeprice_key: str = None,
                  watchcharts_key: str = None,
                  ebay_token: str = None,
                  ebay_sandbox: bool = True) -> dict:
    """
    Run all available connectivity tests.
    
    Args:
        newsapi_key: NewsAPI key (skip test if None)
        brickset_key: Brickset API key (skip test if None)
        pokeprice_key: Pokémon Price Tracker API key (skip test if None)
        watchcharts_key: The Watch API key (skip test if None)
        ebay_token: eBay OAuth token (skip test if None) [DISABLED]
        ebay_sandbox: Use eBay sandbox environment [DISABLED]
        
    Returns:
        Dict with test results:
        {
            "newsapi": {"success": bool, "message": str},
            "brickset": {"success": bool, "message": str},
            "pokeprice": {"success": bool, "message": str},
            "watchcharts": {"success": bool, "message": str},
            "ebay": {"success": None, "message": "not configured"}
        }
    """
    results = {}
    
    # NewsAPI test
    if newsapi_key:
        success, message = test_newsapi(newsapi_key)
        results["newsapi"] = {"success": success, "message": message}
    else:
        results["newsapi"] = {"success": None, "message": "skipped (no API key)"}
    
    # Brickset test
    if brickset_key:
        success, message = test_brickset(brickset_key)
        results["brickset"] = {"success": success, "message": message}
    else:
        results["brickset"] = {"success": None, "message": "skipped (no API key)"}
    
    # Pokémon Price Tracker test
    if pokeprice_key:
        success, message = test_pokeprice(pokeprice_key)
        results["pokeprice"] = {"success": success, "message": message}
    else:
        results["pokeprice"] = {"success": None, "message": "skipped (no API key)"}
    
    # The Watch API test
    if watchcharts_key:
        success, message = test_watchcharts(watchcharts_key)
        results["watchcharts"] = {"success": success, "message": message}
    else:
        results["watchcharts"] = {"success": None, "message": "skipped (no API key)"}
    
    # eBay test (currently disabled)
    results["ebay"] = {"success": None, "message": "not configured yet - coming soon"}
    
    return results
