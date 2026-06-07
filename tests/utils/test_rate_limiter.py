"""
Tests for rate_limiter.py module.
"""
import pytest
import time
from unittest.mock import Mock, patch
from src.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(max_calls=5, time_window=10)
        
        assert limiter.max_calls == 5
        assert limiter.time_window == 10
        assert len(limiter.calls) == 0
    
    def test_rate_limiter_allows_calls_within_limit(self):
        """Test that calls within limit are allowed."""
        limiter = RateLimiter(max_calls=3, time_window=1)
        
        @limiter
        def test_func():
            return "success"
        
        # Should allow 3 calls
        assert test_func() == "success"
        assert test_func() == "success"
        assert test_func() == "success"
        
        assert len(limiter.calls) == 3
    
    def test_rate_limiter_blocks_excess_calls(self):
        """Test that excess calls are blocked/delayed."""
        limiter = RateLimiter(max_calls=2, time_window=1)
        
        call_count = 0
        
        @limiter
        def test_func():
            nonlocal call_count
            call_count += 1
            return call_count
        
        start_time = time.time()
        
        # First 2 calls should be immediate
        test_func()
        test_func()
        
        # Third call should be delayed
        test_func()
        
        elapsed = time.time() - start_time
        
        # Should have taken at least close to 1 second
        assert elapsed >= 0.9
        assert call_count == 3
    
    def test_rate_limiter_cleans_old_calls(self):
        """Test that old calls are removed from the queue."""
        limiter = RateLimiter(max_calls=2, time_window=1)
        
        @limiter
        def test_func():
            return "success"
        
        # Make 2 calls
        test_func()
        test_func()
        
        assert len(limiter.calls) == 2
        
        # Wait for time window to pass
        time.sleep(1.1)
        
        # Make another call - should clean old calls
        test_func()
        
        # Should only have 1 call in queue now
        assert len(limiter.calls) == 1
    
    def test_rate_limiter_with_arguments(self):
        """Test rate limiter with function arguments."""
        limiter = RateLimiter(max_calls=3, time_window=1)
        
        @limiter
        def add(a, b):
            return a + b
        
        assert add(1, 2) == 3
        assert add(5, 10) == 15
        assert add(a=3, b=4) == 7
    
    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        limiter = RateLimiter(max_calls=5, time_window=1)
        
        @limiter
        def test_func():
            return "success"
        
        # The lock should be acquired and released properly
        assert test_func() == "success"
        assert limiter.lock is not None
    
    def test_rate_limiter_multiple_decorators(self):
        """Test multiple functions with different rate limiters."""
        limiter1 = RateLimiter(max_calls=2, time_window=1)
        limiter2 = RateLimiter(max_calls=3, time_window=1)
        
        @limiter1
        def func1():
            return "func1"
        
        @limiter2
        def func2():
            return "func2"
        
        # Each should have independent limits
        assert func1() == "func1"
        assert func1() == "func1"
        
        assert func2() == "func2"
        assert func2() == "func2"
        assert func2() == "func2"
        
        assert len(limiter1.calls) == 2
        assert len(limiter2.calls) == 3
