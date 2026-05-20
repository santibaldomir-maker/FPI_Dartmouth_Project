import os
import gradio_client.utils as _gcu

# gradio_client 4.44.1 crashes when pydantic v2 emits `additionalProperties: True`
# (a Python bool) because it does `if "const" in schema` without a bool check.
_orig_jtype = _gcu._json_schema_to_python_type

def _safe_jtype(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_jtype(schema, defs)

_gcu._json_schema_to_python_type = _safe_jtype

from dtf_platform import demo

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port, show_api=False)
