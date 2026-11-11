"""Run lesson / lab code and capture printed text, matplotlib figures and errors.

This is NOT a sandbox: it uses exec(). That is fine for code you wrote or edit
yourself locally. On a public deployment, editing is disabled by default (see ui.can_edit_code).
"""
from __future__ import annotations

import contextlib
import io
import traceback
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def run_code(code: str) -> dict:
    plt.close("all")
    buf = io.StringIO()
    err = None
    ns = {"__name__": "__academy__"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(compile(code, "<academy-code>", "exec"), ns)
            except (Exception, SystemExit):
                err = traceback.format_exc(limit=-3)
    images = []
    for n in plt.get_fignums():
        b = io.BytesIO()
        plt.figure(n).savefig(b, format="png", dpi=110, bbox_inches="tight")
        images.append(b.getvalue())
    plt.close("all")
    return {"text": buf.getvalue(), "images": images, "error": err}
