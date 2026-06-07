
import time
from collections import deque
from threading import Lock

class RateLimiter:
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = Lock()
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                
                # Remove old calls
                while self.calls and self.calls[0] < now - self.time_window:
                    self.calls.popleft()
                
                # Check limit
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.time_window - (now - self.calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    now = time.time()
                    while self.calls and self.calls[0] < now - self.time_window:
                        self.calls.popleft()
                
                self.calls.append(now)
            
            return func(*args, **kwargs)
        return wrapper

