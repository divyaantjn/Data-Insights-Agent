"""
Comprehensive tests for auth_middleware.py module.
"""
import pytest
import json
import base64
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import Request
from fastapi.responses import JSONResponse
from jwt import ExpiredSignatureError, InvalidTokenError
from middleware.auth_middleware import (
    KeycloakAuthMiddleware,
    get_jwks_uri_for_issuer,
    _b64url_decode,
    JWKS_CACHE,
    JWKS_TTL
)


class TestB64UrlDecode:
    """Tests for _b64url_decode function."""
    
    def test_decode_valid_string(self):
        """Test decoding valid base64url string."""
        # "test" in base64url
        encoded = "dGVzdA"
        result = _b64url_decode(encoded)
        
        assert result == b"test"
    
    def test_decode_with_padding(self):
        """Test decoding string that needs padding."""
        # String that needs padding
        encoded = "dGVzdGluZw"
        result = _b64url_decode(encoded)
        
        assert result == b"testing"
    
    def test_decode_bytes_input(self):
        """Test decoding bytes input."""
        encoded = b"dGVzdA"
        result = _b64url_decode(encoded)
        
        assert result == b"test"
    
    def test_decode_empty_string(self):
        """Test decoding empty string."""
        result = _b64url_decode("")
        
        assert result == b""
    
    def test_decode_with_special_chars(self):
        """Test decoding with URL-safe characters."""
        # Base64url uses - and _ instead of + and /
        encoded = "SGVsbG8gV29ybGQh"
        result = _b64url_decode(encoded)
        
        assert b"Hello World!" in result or result == b"Hello World!"


class TestGetJwksUriForIssuer:
    """Tests for get_jwks_uri_for_issuer function."""
    
    @pytest.mark.asyncio
    async def test_get_jwks_uri_from_cache(self):
        """Test getting JWKS URI from cache."""
        issuer = "https://test.keycloak.com/realms/test"
        jwks_uri = "https://test.keycloak.com/realms/test/protocol/openid-connect/certs"
        
        # Populate cache
        JWKS_CACHE[issuer] = {
            "jwks_uri": jwks_uri,
            "fetched_at": time.time()
        }
        
        result = await get_jwks_uri_for_issuer(issuer)
        
        assert result == jwks_uri
    
    @pytest.mark.asyncio
    async def test_get_jwks_uri_cache_expired(self):
        """Test getting JWKS URI when cache is expired."""
        issuer = "https://test.keycloak.com/realms/test"
        jwks_uri = "https://test.keycloak.com/realms/test/protocol/openid-connect/certs"
        
        # Populate cache with expired entry
        JWKS_CACHE[issuer] = {
            "jwks_uri": "old_uri",
            "fetched_at": time.time() - JWKS_TTL - 100
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jwks_uri": jwks_uri}
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await get_jwks_uri_for_issuer(issuer)
            
            assert result == jwks_uri
            assert JWKS_CACHE[issuer]["jwks_uri"] == jwks_uri
    
    @pytest.mark.asyncio
    async def test_get_jwks_uri_not_in_cache(self):
        """Test getting JWKS URI when not in cache."""
        issuer = "https://new.keycloak.com/realms/test"
        jwks_uri = "https://new.keycloak.com/realms/test/protocol/openid-connect/certs"
        
        # Clear cache for this issuer
        JWKS_CACHE.pop(issuer, None)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jwks_uri": jwks_uri}
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await get_jwks_uri_for_issuer(issuer)
            
            assert result == jwks_uri
            assert issuer in JWKS_CACHE
    
    @pytest.mark.asyncio
    async def test_get_jwks_uri_discovery_fails(self):
        """Test getting JWKS URI when discovery endpoint fails."""
        issuer = "https://fail.keycloak.com/realms/test"
        
        JWKS_CACHE.pop(issuer, None)
        
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            with pytest.raises(Exception):
                await get_jwks_uri_for_issuer(issuer)


class TestKeycloakAuthMiddlewarePublicPaths:
    """Tests for public path handling in KeycloakAuthMiddleware."""
    
    @pytest.mark.asyncio
    async def test_public_path_root(self):
        """Test that root path is public."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        
        mock_response = JSONResponse({"status": "ok"})
        call_next = AsyncMock(return_value=mock_response)
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 200
        call_next.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_public_path_health(self):
        """Test that /health path is public."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/health"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        
        mock_response = JSONResponse({"status": "healthy"})
        call_next = AsyncMock(return_value=mock_response)
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 200
        call_next.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_public_path_docs(self):
        """Test that /docs path is public."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/docs"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        
        mock_response = JSONResponse({"docs": "page"})
        call_next = AsyncMock(return_value=mock_response)
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_public_path_openapi(self):
        """Test that /openapi.json path is public."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/openapi.json"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        
        mock_response = JSONResponse({"openapi": "3.0.0"})
        call_next = AsyncMock(return_value=mock_response)
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 200


class TestKeycloakAuthMiddlewareAuthRequired:
    """Tests for authentication required paths."""
    
    @pytest.mark.asyncio
    async def test_missing_authorization_header(self):
        """Test request without Authorization header."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {}
        
        call_next = AsyncMock()
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 401
        assert "Missing Authorization Header" in str(response.body)
        call_next.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_invalid_authorization_scheme(self):
        """Test request with invalid authorization scheme."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": "Basic dGVzdDp0ZXN0"}
        
        call_next = AsyncMock()
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 401
        assert "Invalid Authorization Header" in str(response.body)
    
    @pytest.mark.asyncio
    async def test_empty_token(self):
        """Test request with empty token."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": "Bearer "}
        
        call_next = AsyncMock()
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_token_with_custom_separator(self):
        """Test token with custom separator."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        # Create a token with custom separator
        token_part = "header.payload.signature"
        full_token = f"{token_part}$YashUnified2025$extra_data"
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": f"Bearer {full_token}"}
        
        call_next = AsyncMock()
        
        # Should extract only the first part before separator
        response = await middleware.dispatch(request, call_next)
        
        # Will fail validation but should process the token part
        assert response.status_code in [400, 401, 403]


class TestKeycloakAuthMiddlewareTokenValidation:
    """Tests for token validation."""
    
    @pytest.mark.asyncio
    async def test_token_invalid_parts(self):
        """Test token with invalid number of parts."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": "Bearer invalid.token"}
        
        call_next = AsyncMock()
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 400
        assert "3 parts" in str(response.body)
    
    @pytest.mark.asyncio
    async def test_token_invalid_payload_encoding(self):
        """Test token with invalid payload encoding."""
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        # Token with invalid base64 in payload
        request.headers = {"Authorization": "Bearer header.!!!invalid!!!.signature"}
        
        call_next = AsyncMock()
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'KEYCLOAK_ISSUER': 'https://keycloak.test.com/realms'})
    async def test_token_untrusted_issuer(self):
        """Test token with untrusted issuer."""
        # Reload the module to pick up the patched environment variables
        import importlib
        import middleware.auth_middleware
        importlib.reload(middleware.auth_middleware)
        from middleware.auth_middleware import KeycloakAuthMiddleware
        
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        # Create a payload with untrusted issuer
        payload = {"iss": "https://evil.com/realms/test"}
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
        
        token = f"header.{payload_b64}.signature"
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": f"Bearer {token}"}
        
        call_next = AsyncMock()
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 403
        assert "Untrusted token issuer" in str(response.body)
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {
        'KEYCLOAK_ISSUER': 'https://keycloak.test.com/realms',
        'KEYCLOAK_CLIENT_ID': 'test-client'
    })
    async def test_token_expired(self):
        """Test expired token."""
        # Reload the module to pick up the patched environment variables
        import importlib
        import middleware.auth_middleware
        importlib.reload(middleware.auth_middleware)
        from middleware.auth_middleware import KeycloakAuthMiddleware
        
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        # Create a payload with trusted issuer
        payload = {"iss": "https://keycloak.test.com/realms/test"}
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
        
        token = f"header.{payload_b64}.signature"
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": f"Bearer {token}"}
        request.state = Mock()
        
        call_next = AsyncMock()
        
        with patch('middleware.auth_middleware.get_jwks_uri_for_issuer', new_callable=AsyncMock) as mock_jwks:
            mock_jwks.return_value = "https://keycloak.test.com/certs"
            
            with patch('jwt.PyJWKClient') as mock_jwk_client:
                mock_client = Mock()
                mock_signing_key = Mock()
                mock_signing_key.key = "test_key"
                mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
                mock_jwk_client.return_value = mock_client
                
                with patch('jwt.decode', side_effect=ExpiredSignatureError("Token expired")):
                    response = await middleware.dispatch(request, call_next)
                    
                    assert response.status_code == 401
                    assert "expired" in str(response.body).lower()
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {
        'KEYCLOAK_ISSUER': 'https://keycloak.test.com/realms',
        'KEYCLOAK_CLIENT_ID': 'test-client'
    })
    async def test_token_invalid(self):
        """Test invalid token."""
        # Reload the module to pick up the patched environment variables
        import importlib
        import middleware.auth_middleware
        importlib.reload(middleware.auth_middleware)
        from middleware.auth_middleware import KeycloakAuthMiddleware
        
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        payload = {"iss": "https://keycloak.test.com/realms/test"}
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
        
        token = f"header.{payload_b64}.signature"
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": f"Bearer {token}"}
        request.state = Mock()
        
        call_next = AsyncMock()
        
        with patch('middleware.auth_middleware.get_jwks_uri_for_issuer', new_callable=AsyncMock) as mock_jwks:
            mock_jwks.return_value = "https://keycloak.test.com/certs"
            
            with patch('jwt.PyJWKClient') as mock_jwk_client:
                mock_client = Mock()
                mock_signing_key = Mock()
                mock_signing_key.key = "test_key"
                mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
                mock_jwk_client.return_value = mock_client
                
                with patch('jwt.decode', side_effect=InvalidTokenError("Invalid token")):
                    response = await middleware.dispatch(request, call_next)
                    
                    assert response.status_code == 401
                    assert "Invalid token" in str(response.body)


class TestKeycloakAuthMiddlewareSuccessfulAuth:
    """Tests for successful authentication."""
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {
        'KEYCLOAK_ISSUER': 'https://keycloak.test.com/realms',
        'KEYCLOAK_CLIENT_ID': 'test-client'
    })
    async def test_successful_authentication(self):
        """Test successful authentication with valid token."""
        # Reload the module to pick up the patched environment variables
        import importlib
        import middleware.auth_middleware
        importlib.reload(middleware.auth_middleware)
        from middleware.auth_middleware import KeycloakAuthMiddleware
        
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        payload = {"iss": "https://keycloak.test.com/realms/test"}
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
        
        token = f"header.{payload_b64}.signature"
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": f"Bearer {token}"}
        request.state = Mock()
        request.method = "GET"
        request.client.host = "127.0.0.1"
        
        decoded_payload = {
            "iss": "https://keycloak.test.com/realms/test",
            "sub": "user123",
            "azp": "test-client",
            "preferred_username": "testuser",
            "resource_access": {
                "test-client": {
                    "roles": ["test-client_client"]
                }
            }
        }
        
        mock_response = JSONResponse({"data": "protected"})
        call_next = AsyncMock(return_value=mock_response)
        
        with patch('middleware.auth_middleware.get_jwks_uri_for_issuer', new_callable=AsyncMock) as mock_jwks:
            mock_jwks.return_value = "https://keycloak.test.com/certs"
            
            with patch('jwt.PyJWKClient') as mock_jwk_client:
                mock_client = Mock()
                mock_signing_key = Mock()
                mock_signing_key.key = "test_key"
                mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
                mock_jwk_client.return_value = mock_client
                
                with patch('jwt.decode', return_value=decoded_payload):
                    response = await middleware.dispatch(request, call_next)
                    
                    assert response.status_code == 200
                    assert request.state.user == decoded_payload
                    call_next.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {
        'KEYCLOAK_ISSUER': 'https://keycloak.test.com/realms',
        'KEYCLOAK_CLIENT_ID': 'test-client'
    })
    async def test_missing_required_role(self):
        """Test authentication with missing required role."""
        # Reload the module to pick up the patched environment variables
        import importlib
        import middleware.auth_middleware
        importlib.reload(middleware.auth_middleware)
        from middleware.auth_middleware import KeycloakAuthMiddleware
        
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        payload = {"iss": "https://keycloak.test.com/realms/test"}
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
        
        token = f"header.{payload_b64}.signature"
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": f"Bearer {token}"}
        request.state = Mock()
        
        decoded_payload = {
            "iss": "https://keycloak.test.com/realms/test",
            "sub": "user123",
            "azp": "test-client",
            "resource_access": {
                "test-client": {
                    "roles": ["other_role"]  # Missing required role
                }
            }
        }
        
        call_next = AsyncMock()
        
        with patch('middleware.auth_middleware.get_jwks_uri_for_issuer', new_callable=AsyncMock) as mock_jwks:
            mock_jwks.return_value = "https://keycloak.test.com/certs"
            
            with patch('jwt.PyJWKClient') as mock_jwk_client:
                mock_client = Mock()
                mock_signing_key = Mock()
                mock_signing_key.key = "test_key"
                mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
                mock_jwk_client.return_value = mock_client
                
                with patch('jwt.decode', return_value=decoded_payload):
                    response = await middleware.dispatch(request, call_next)
                    
                    assert response.status_code == 400
                    assert "Missing required role" in str(response.body)
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {
        'KEYCLOAK_ISSUER': 'https://keycloak.test.com/realms',
        'KEYCLOAK_CLIENT_ID': 'test-client'
    })
    async def test_missing_client_in_resource_access(self):
        """Test authentication with missing client in resource_access."""
        # Reload the module to pick up the patched environment variables
        import importlib
        import middleware.auth_middleware
        importlib.reload(middleware.auth_middleware)
        from middleware.auth_middleware import KeycloakAuthMiddleware
        
        app = Mock()
        middleware = KeycloakAuthMiddleware(app)
        
        payload = {"iss": "https://keycloak.test.com/realms/test"}
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
        
        token = f"header.{payload_b64}.signature"
        
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.headers = {"Authorization": f"Bearer {token}"}
        request.state = Mock()
        
        decoded_payload = {
            "iss": "https://keycloak.test.com/realms/test",
            "sub": "user123",
            "azp": "test-client",
            "resource_access": {
                "other-client": {
                    "roles": ["some_role"]
                }
            }
        }
        
        call_next = AsyncMock()
        
        with patch('middleware.auth_middleware.get_jwks_uri_for_issuer', new_callable=AsyncMock) as mock_jwks:
            mock_jwks.return_value = "https://keycloak.test.com/certs"
            
            with patch('jwt.PyJWKClient') as mock_jwk_client:
                mock_client = Mock()
                mock_signing_key = Mock()
                mock_signing_key.key = "test_key"
                mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
                mock_jwk_client.return_value = mock_client
                
                with patch('jwt.decode', return_value=decoded_payload):
                    response = await middleware.dispatch(request, call_next)
                    
                    assert response.status_code == 400
                    assert "not found in resource_access" in str(response.body)
