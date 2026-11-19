from functools import wraps

from flask import jsonify, session


def require_login(view):
    """Require a server-signed Flask session; never trust client-supplied user IDs."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        user_id = session.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            return jsonify({"error": "Authentication required. Please sign in again."}), 401
        return view(*args, **kwargs)
    return wrapped
