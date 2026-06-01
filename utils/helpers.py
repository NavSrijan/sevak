import uuid

def to_uuid(session_id: str) -> str:
    """Convert a session ID to a UUID string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id))
