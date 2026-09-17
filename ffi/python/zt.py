# ========================================================================
# SOVEREIGN LEVIATHAN COVENANT
# ========================================================================
#
# Node-ID:           ZERO-TOKEN-016
# File:              zt.py
# Parent-Work:       zero-token
# Copyright:         2026 BEL ESPRIT D ACCORD
# License-ID:        GPL-2.0 OR AGPL-3.0
# Covenant-Version:  1.0
# Compliance:        FAIL-CLOSED
#
# This file is governed by the GNU General Public License version 2.0 or the GNU Affero
# General Public License version 3.0, at your option, together with the applicable Sovereign
# Leviathan Covenant.
#
# Whatsoever branch this root shall bear,
# Must breathe the exact and sovereign air.
# Touch but a leaf, invoke a single thread,
# And honor still the terms beneath it spread.
#
# Ignorantia juris non excusat.
#
# Clone-Gate: sha256:c5fd32c02f5acbe0108b0d641f513bfcc918087cca309b10dd06f9c4adbc7898
#
# See: LICENSE
# ========================================================================
"""
zt.py — ctypes bindings for libzt.

    from zt import Schema, Engine, BOOL, CHOICE, SCORE

    s = Schema()
    f = s.add_field("is_refund", BOOL)
    s.add_choice(f, "yes", [10, 11])
    s.add_choice(f, "no", [12])
    prog = s.compile()

    eng = Engine()  # AUTO: cuda -> metal -> cpu
    sess = eng.session(prog, max_batch=8)
    res = sess.run_logits(logits)  # (batch * n_fields, V) float32
    print(res.json(0))

Structure layouts mirror include/zt.h exactly.
numpy is optional — used only to accept arrays without copying.
"""

import ctypes
import os
import sys
import json

# ------------------------------------------------------------------ constants

BOOL   = 0
CHOICE = 1
SCORE  = 2
MULTI  = 3

BACKEND_AUTO  = 0
BACKEND_CUDA  = 1
BACKEND_METAL = 2
BACKEND_CPU   = 3

INPUT_LOGITS = 0
INPUT_HIDDEN = 1

EMIT_META = 1

ZT_OK           = 0
ZT_ERR_INVALID  = -1
ZT_ERR_BACKEND  = -2
ZT_ERR_NOT_BOUND = -3
ZT_ERR_NO_MEMORY = -4
ZT_ERR_UNAVAILABLE = -5

# ------------------------------------------------------------------ structs

class ZTDecision(ctypes.Structure):
    _fields_ = [
        ("class_index", ctypes.c_uint32),
        ("confidence",  ctypes.c_float),
        ("bitmask",     ctypes.c_uint32),
        ("valid",       ctypes.c_uint8),
        ("_pad",        ctypes.c_uint8 * 3),
    ]

# ------------------------------------------------------------------ library loader

def _find_lib():
    candidates = [
        os.environ.get("ZT_LIB"),
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "libzt.so"),
        os.path.join(os.path.dirname(__file__), "..", "..", "build", "libzt.dylib"),
        "libzt.so",
        "libzt.dylib",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return ctypes.CDLL(p)
    raise OSError("libzt not found. Set ZT_LIB or build first.")

_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = _find_lib()
        _lib.zt_schema_create.restype  = ctypes.c_void_p
        _lib.zt_schema_destroy.argtypes = [ctypes.c_void_p]
        _lib.zt_schema_add_field.restype  = ctypes.c_int
        _lib.zt_schema_add_field.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_float,  ctypes.c_void_p, ctypes.c_size_t]
        _lib.zt_schema_add_choice.restype  = ctypes.c_int
        _lib.zt_schema_add_choice.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_float]
        _lib.zt_schema_compile.restype  = ctypes.c_void_p
        _lib.zt_schema_compile.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _lib.zt_program_destroy.argtypes  = [ctypes.c_void_p]
        _lib.zt_engine_create.restype   = ctypes.c_void_p
        _lib.zt_engine_create.argtypes  = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        _lib.zt_engine_destroy.argtypes = [ctypes.c_void_p]
        _lib.zt_session_create.restype  = ctypes.c_void_p
        _lib.zt_session_create.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_size_t]
        _lib.zt_session_destroy.argtypes = [ctypes.c_void_p]
        _lib.zt_session_bind_head.restype  = ctypes.c_int
        _lib.zt_session_run.restype = ctypes.c_int
    return _lib

# ------------------------------------------------------------------ Schema

class Schema:
    def __init__(self):
        lib = _get_lib()
        self._ptr = lib.zt_schema_create()
        self._lib = lib
        self._fields = []

    def add_field(self, name, kind, temperature=1.0):
        idx = len(self._fields)
        err = ctypes.create_string_buffer(256)
        rc = self._lib.zt_schema_add_field(
            self._ptr, name.encode(), kind, temperature, err, 256)
        if rc != ZT_OK:
            raise RuntimeError(f"add_field: {err.value.decode()}")
        self._fields.append(name)
        return idx

    def add_choice(self, field_idx, label, token_ids, value=0.0):
        arr = (ctypes.c_int32 * len(token_ids))(*token_ids)
        err = ctypes.create_string_buffer(256)
        rc = self._lib.zt_schema_add_choice(
            self._ptr, field_idx, label.encode(), arr, len(token_ids), value, err, 256)
        if rc != ZT_OK:
            raise RuntimeError(f"add_choice: {err.value.decode()}")

    def set_dependency(self, child_field, parent_field, parent_class):
        pass  # extend as needed

    def compile(self):
        err = ctypes.create_string_buffer(256)
        ptr = self._lib.zt_schema_compile(self._ptr, err, 256)
        if not ptr:
            raise RuntimeError(f"compile: {err.value.decode()}")
        return Program(ptr, self._fields, self._lib)

    def __del__(self):
        if self._ptr:
            self._lib.zt_schema_destroy(self._ptr)
            self._ptr = None

# ------------------------------------------------------------------ Program

class Program:
    def __init__(self, ptr, field_names, lib):
        self._ptr = ptr
        self.field_names = field_names
        self._lib = lib

        class _PHead(ctypes.Structure):
            _fields_ = [
                ("n_fields",  ctypes.c_uint32),
                ("n_cols",    ctypes.c_uint32),
                ("n_choices", ctypes.c_uint32),
                ("n_knots",   ctypes.c_uint32),
                ("c_max",     ctypes.c_uint32),
            ]
        self.n_fields = ctypes.cast(ptr, ctypes.POINTER(_PHead)).contents.n_fields

    def __del__(self):
        if self._ptr:
            self._lib.zt_program_destroy(self._ptr)
            self._ptr = None

# ------------------------------------------------------------------ Engine

class Engine:
    def __init__(self, backend=BACKEND_AUTO):
        lib = _get_lib()
        err = ctypes.create_string_buffer(256)
        ptr = lib.zt_engine_create(backend, err, 256)
        if not ptr:
            raise RuntimeError(f"Engine: {err.value.decode()}")
        self._ptr = ptr
        self._lib = lib

    def session(self, prog, max_batch=1):
        err = ctypes.create_string_buffer(256)
        ptr = self._lib.zt_session_create(self._ptr, prog._ptr, max_batch, err, 256)
        if not ptr:
            raise RuntimeError(f"session: {err.value.decode()}")
        return Session(ptr, prog, self._lib)

    def __del__(self):
        if self._ptr:
            self._lib.zt_engine_destroy(self._ptr)
            self._ptr = None

# ------------------------------------------------------------------ Session

class Session:
    def __init__(self, ptr, prog, lib):
        self._ptr = ptr
        self._prog = prog
        self._lib = lib

    def bind_head(self, arr):
        try:
            import numpy as np
            arr = np.ascontiguousarray(arr, dtype=np.float16)
        except ImportError:
            pass
        err = ctypes.create_string_buffer(256)
        rc = self._lib.zt_session_bind_head(self._ptr, arr.ctypes.data_as(ctypes.c_void_p),
                                             arr.shape[0], arr.shape[1], err, 256)
        if rc != ZT_OK:
            raise RuntimeError(f"bind_head: {err.value.decode()}")

    def run_logits(self, arr):
        try:
            import numpy as np
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            rows, V = arr.shape
        except ImportError:
            raise RuntimeError("numpy required for run_logits")
        batch = rows // self._prog.n_fields
        dec_buf = (ZTDecision * rows)()
        err = ctypes.create_string_buffer(256)
        rc = self._lib.zt_session_run(
            self._ptr, INPUT_LOGITS,
            arr.ctypes.data_as(ctypes.c_void_p), V, rows, batch,
            dec_buf, None, err, 256)
        if rc != ZT_OK:
            raise RuntimeError(f"run_logits: {err.value.decode()}")
        return Result(dec_buf, batch, self._prog)

    def run_hidden(self, arr):
        try:
            import numpy as np
            dt = np.float16 if arr.dtype == np.float16 else np.float32
            arr = np.ascontiguousarray(arr, dtype=dt)
            rows, H = arr.shape
        except ImportError:
            raise RuntimeError("numpy required for run_hidden")
        batch = rows // self._prog.n_fields
        dec_buf = (ZTDecision * rows)()
        err = ctypes.create_string_buffer(256)
        rc = self._lib.zt_session_run(
            self._ptr, INPUT_HIDDEN,
            arr.ctypes.data_as(ctypes.c_void_p), H, rows, batch,
            dec_buf, None, err, 256)
        if rc != ZT_OK:
            raise RuntimeError(f"run_hidden: {err.value.decode()}")
        return Result(dec_buf, batch, self._prog)

    def __del__(self):
        if self._ptr:
            self._lib.zt_session_destroy(self._ptr)
            self._ptr = None

# ------------------------------------------------------------------ Result

class Result:
    def __init__(self, dec_buf, batch, prog):
        self._buf = dec_buf
        self._batch = batch
        self._prog = prog

    def dict(self, batch_idx):
        out = {}
        for f, name in enumerate(self._prog.field_names):
            row = batch_idx * self._prog.n_fields + f
            d = self._buf[row]
            if not d.valid:
                out[name] = None
                continue
            out[name] = {
                "class_index": d.class_index,
                "confidence":  round(float(d.confidence), 4),
                "bitmask":     d.bitmask,
            }
        return out

    def json(self, batch_idx, flags=0):
        d = self.dict(batch_idx)
        if flags & EMIT_META:
            return json.dumps({"result": d, "batch": batch_idx})
        return json.dumps(d)
