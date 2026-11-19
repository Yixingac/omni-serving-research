"""Session-level load drivers for the pre-presentation validation (R3 camera replay, R4 robot sessions).

They run as ``plugin`` phases of the experiment runner (``{"kind": "plugin", "module": "sessions.camera", ...}``)
and use only the standard library, like the rest of the client side.
"""
