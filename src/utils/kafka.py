# src/utils/kafka.py

import os
import json
import logging
from typing import Optional

from .kafka_base import KafkaManager, extract_user_context

logger = logging.getLogger(__name__)


class BaseKafkaLogger:
    """Base class for Kafka loggers with shared sending logic."""
    
    def __init__(self, topic: str, debug_prefix: str):
        self.topic = topic
        self.debug_prefix = debug_prefix
        self.kafka_manager = KafkaManager()
    
    def _on_send_success(self, record_metadata):
        """Callback for successful message sends."""
        logger.debug(f"Message delivered to topic '{record_metadata.topic}' partition {record_metadata.partition}")
    
    def _on_send_error(self, excp):
        """Callback for failed message sends."""
        logger.error(f"Error sending message to Kafka: {excp}", exc_info=excp)
    
    def _send(self, payload: dict, show_debug: bool = True):
        """
        Generic send method that uses the Kafka Manager singleton.
        
        Args:
            payload: Dictionary payload to send
            show_debug: Whether to print debug output
        """
        try:
            if show_debug:
                print(f"\n--- [{self.debug_prefix}] ---")
                print(json.dumps(payload, indent=2))
                print(f"{'-' * (len(self.debug_prefix) + 10)}\n")
        except Exception as e:
            print(f"--- [{self.debug_prefix}] FAILED: {e} ---")
        
        producer = self.kafka_manager.get_producer()
        if not producer:
            logger.warning(f"Kafka producer not available. Message not sent to topic '{self.topic}'.")
            return
        
        try:
            # Non-blocking send with timeout protection
            future = producer.send(self.topic, value=payload)
            # Don't wait for the result - fire and forget
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
        except KeyboardInterrupt:
            raise  # Allow keyboard interrupt to propagate
        except Exception as e:
            # Catch all exceptions to prevent Kafka from interrupting server flow
            error_msg = str(e).lower()
            if "timeout" in error_msg or "not found" in error_msg or "max_block_ms" in error_msg or "kafka" in error_msg:
                logger.debug(f"Kafka message not sent to '{self.topic}': {error_msg}")
            else:
                logger.warning(f"Non-critical Kafka error for topic '{self.topic}': {e}")


class KafkaLogger(BaseKafkaLogger):
    """Token usage logger that sends payloads to Kafka."""
    
    def __init__(self):
        topic = os.getenv("KAFKA_TOPIC_NAME", "llm-token-usage-default")
        super().__init__(topic=topic, debug_prefix="KAFKA PAYLOAD DEBUG")
    
    def log(self, data: dict):
        """Send token usage data to Kafka."""
        try:
            self._send(data, show_debug=True)
        except Exception as e:
            logger.debug(f"Failed to log token usage to Kafka: {e}")
    
    def close(self):
        """Close the Kafka producer (delegates to singleton manager)."""
        try:
            self.kafka_manager.close()
        except Exception as e:
            logger.debug(f"Error closing Kafka logger: {e}")


class KafkaEventLogger(BaseKafkaLogger):
    """Event Logger for real-time user visibility via Kafka."""
    
    def __init__(self):
        topic = os.getenv("KAFKA_EVENT_TOPIC_NAME", "agent-event-notification")
        super().__init__(topic=topic, debug_prefix="KAFKA EVENT DEBUG")
        self.agent_name = os.getenv("AGENT_NAME", "DATA_INSIGHTS_AGENT")
        self.server_name = os.getenv("SERVER_NAME", "DATA_INSIGHTS_BACKEND")
    
    def _create_base_event(self, message: str, auth_token: Optional[str] = None) -> dict:
        """Create base event with encrypted_payload, timestamp, message."""
        from datetime import datetime, timezone
        user_context = extract_user_context(auth_token)
        return {
            "encrypted_payload": user_context["encrypted_payload"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "type": "agent-event",
            "kafka_topic_name": self.topic,
            "server_name": self.server_name
        }
    
    def log_event(self, message: str, auth_token: Optional[str] = None):
        """Log a general event."""
        try:
            event = self._create_base_event(message, auth_token)
            show_debug = event.get("encrypted_payload") != "NO_PAYLOAD"
            self._send(event, show_debug=show_debug)
        except Exception as e:
            logger.debug(f"Failed to log event to Kafka: {e}")
    
    def close(self):
        """Close the event logger producer (delegates to singleton manager)."""
        try:
            self.kafka_manager.close()
        except Exception as e:
            logger.debug(f"Error closing Kafka event logger: {e}")


def create_event_logger() -> KafkaEventLogger:
    """Create a new event logger instance."""
    return KafkaEventLogger()

class ReasoningLogger(BaseKafkaLogger):
    """Reasoning Logger for LLM thinking/reasoning to Kafka."""
    
    def __init__(self):
        topic = os.getenv("KAFKA_REASONING_TOPIC_NAME", "agent-reasoning-notification")
        super().__init__(topic=topic, debug_prefix="KAFKA REASONING PAYLOAD")
        self.server_name = os.getenv("SERVER_NAME", "DATA_INSIGHTS_BACKEND")
    
    def log_reasoning(self, reasoning: str, auth_token: Optional[str] = None):
        """Send reasoning event to Kafka."""
        from datetime import datetime, timezone
        try:
            user_context = extract_user_context(auth_token)
            payload = {
                "encrypted_payload": user_context["encrypted_payload"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": reasoning,
                "kafka_topic_name": self.topic,
                "server_name": self.server_name,
                "type": "agent-reasoning"
            }
            self._send(payload, show_debug=True)
        except Exception as e:
            logger.debug(f"Failed to log reasoning to Kafka: {e}")
    
    def close(self):
        try:
            self.kafka_manager.close()
        except Exception as e:
            logger.debug(f"Error closing Kafka reasoning logger: {e}")


def create_reasoning_logger() -> ReasoningLogger:
    """Create a new reasoning logger instance."""
    return ReasoningLogger()


# Global instances
kafka_logger = KafkaLogger()

