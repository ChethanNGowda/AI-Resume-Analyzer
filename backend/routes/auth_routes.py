from datetime import datetime, timedelta
import os
import re

from flask import Blueprint, jsonify, request, session
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint("auth", __name__)

# Runtime-only account store: passwords are hashed and no demo credentials ship publicly.
# Render Free has ephemeral storage; accounts reset when the service restarts or redeploys.
USERS = {}

# Seed a default dev account from env vars if provided (keeps session alive across restarts)
_DEFAULT_USER = os.getenv("DEFAULT_USERNAME", "")
_DEFAULT_PASS = os.getenv("DEFAULT_PASSWORD", "")
if _DEFAULT_USER and _DEFAULT_PASS:
    USERS[_DEFAULT_USER] = {
        "password": generate_password_hash(_DEFAULT_PASS),
        "role": "user",
        "subscription": "free",
        "email": f"{_DEFAULT_USER}@local.dev",
    }


def public_profile(username, user):
    display_name = username.replace("_", " ").replace(".", " ").title()
    initials = "".join(part[0] for part in display_name.split()[:2]).upper() or username[:2].upper()
    return {
        "success": True,
        "username": username,
        "display_name": display_name,
        "initials": initials[:2],
        "email": user["email"],
        "role": user["role"],
        "subscription": user["subscription"],
        "github_username": user.get("github_username"),
        "profile_verified": True,
        "verified_by": "server-session",
    }


@auth_bp.route("/register", methods=["POST"])
def register():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))

    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        return jsonify({"error": "Username must be 3–32 characters and use only letters, numbers, dots, underscores, or hyphens."}), 400
    if len(password) < 12:
        return jsonify({"error": "Password must be at least 12 characters long."}), 400
    if username.lower() in (name.lower() for name in USERS):
        return jsonify({"error": "Username already exists."}), 409

    USERS[username] = {
        "password": generate_password_hash(password),
        "role": "user",
        "subscription": "free",
        "email": f"{username}@example.invalid",
    }
    return jsonify({"success": True, "message": "Account created. Please sign in.", "username": username}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))

    matched_username = next((name for name in USERS if name.lower() == username.lower()), None)
    user = USERS.get(matched_username) if matched_username else None
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid username or password."}), 401

    session.clear()
    session.permanent = True
    session["user_id"] = matched_username
    return jsonify({**public_profile(matched_username, user), "loginTime": datetime.now().isoformat()})


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Signed out."})


@auth_bp.route("/profile/<username>", methods=["GET"])
def profile(username):
    current_user = session.get("user_id")
    if not current_user or current_user.lower() != username.strip().lower():
        return jsonify({"error": "Authentication required.", "profile_verified": False}), 401
    user = USERS.get(current_user)
    if not user:
        session.clear()
        return jsonify({"error": "Session expired. Please register again.", "profile_verified": False}), 401
    return jsonify(public_profile(current_user, user))


@auth_bp.route("/subscribe", methods=["POST"])
def subscribe():
    # Plan changes must be backed by a real payment/entitlement provider.
    return jsonify({"error": "Subscription changes are not enabled on this deployment."}), 403
