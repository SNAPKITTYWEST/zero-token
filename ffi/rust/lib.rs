// ========================================================================
// SOVEREIGN LEVIATHAN COVENANT
// ========================================================================
//
// Node-ID:           ZERO-TOKEN-017
// File:              lib.rs
// Parent-Work:       zero-token
// Copyright:         2026 BEL ESPRIT D ACCORD
// License-ID:        GPL-2.0 OR AGPL-3.0
// Covenant-Version:  1.0
// Compliance:        FAIL-CLOSED
//
// This file is governed by the GNU General Public License version 2.0 or the GNU Affero
// General Public License version 3.0, at your option, together with the applicable Sovereign
// Leviathan Covenant.
//
// Whatsoever branch this root shall bear,
// Must breathe the exact and sovereign air.
// Touch but a leaf, invoke a single thread,
// And honor still the terms beneath it spread.
//
// Ignorantia juris non excusat.
//
// Clone-Gate: sha256:6049f9a4f2f043404f612d88695b0b83f4bdd8ec0700b3fa4fc17a0d086f753a
//
// See: LICENSE
// ========================================================================
//!
//! Rust bindings for the zero-token decision and routing engine.
//!
//! ```no_run
//! use zero_token_sys::{Schema, Engine, ResultBuf, Kind, Backend};
//!
//! let mut s = Schema::new().unwrap();
//! let f = s.add_field("is_refund", Kind::Bool).unwrap();
//! s.add_choice(f, "yes", &[10, 11], 0.0).unwrap();
//! s.add_choice(f, "no", &[12], 0.0).unwrap();
//! let prog = s.compile().unwrap();
//!
//! let engine = Engine::new(Backend::Cpu).unwrap();
//! let mut sess = engine.session(&prog, 1).unwrap();
//! let mut out = ResultBuf::new(&prog, 1, false).unwrap();
//!
//! // One row per (batch item, field), row-major logits over vocabulary V.
//! let logits = vec![0.0f32; 16];
//! sess.run_logits(&logits, 16, &mut out).unwrap();
//! println!("{}", out.json(0, 0).unwrap());
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

// ------------------------------------------------------------------ ABI types

#[repr(C)]
pub struct ZtDecision {
    pub class_index: u32,
    pub confidence:  f32,
    pub bitmask:     u32,
    pub valid:       u8,
    _pad:            [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Kind { Bool = 0, Choice = 1, Score = 2, Multi = 3 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Backend { Auto = 0, Cuda = 1, Metal = 2, Cpu = 3 }

#[derive(Debug)]
pub struct Error(String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "{}", self.0) }
}
impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

// ------------------------------------------------------------------ extern

extern "C" {
    fn zt_schema_create() -> *mut c_void;
    fn zt_schema_destroy(s: *mut c_void);
    fn zt_schema_add_field(s: *mut c_void, name: *const c_char, kind: c_int, temperature: f32,
                           err: *mut c_char, errcap: usize) -> c_int;
    fn zt_schema_add_choice(s: *mut c_void, field: c_int, label: *const c_char,
                            tokens: *const i32, n: u32, value: f32,
                            err: *mut c_char, errcap: usize) -> c_int;
    fn zt_schema_compile(s: *mut c_void, err: *mut c_char, errcap: usize) -> *mut c_void;
    fn zt_program_destroy(p: *mut c_void);
    fn zt_program_n_fields(p: *const c_void) -> u32;
    fn zt_engine_create(backend: c_int, err: *mut c_char, errcap: usize) -> *mut c_void;
    fn zt_engine_destroy(e: *mut c_void);
    fn zt_session_create(e: *mut c_void, p: *const c_void, max_batch: u32,
                         err: *mut c_char, errcap: usize) -> *mut c_void;
    fn zt_session_destroy(s: *mut c_void);
    fn zt_session_bind_head(s: *mut c_void, data: *const c_void,
                             rows: u32, cols: u32,
                             err: *mut c_char, errcap: usize) -> c_int;
    fn zt_session_run(s: *mut c_void, kind: c_int,
                      data: *const c_void, ld: u32, rows: u32, batch: u32,
                      dec: *mut ZtDecision, probs: *mut f32,
                      err: *mut c_char, errcap: usize) -> c_int;
    fn zt_result_json(dec: *const ZtDecision, field_names: *const *const c_char,
                      n_fields: u32, batch_idx: u32, flags: u32,
                      buf: *mut c_char, bufsz: usize) -> c_int;
}

fn errbuf() -> [c_char; 512] { [0; 512] }
fn check(rc: c_int, err: &[c_char]) -> Result<()> {
    if rc == 0 { Ok(()) }
    else {
        let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
        Err(Error(msg))
    }
}

// ------------------------------------------------------------------ Schema

pub struct Schema { ptr: *mut c_void, field_names: Vec<CString> }

impl Schema {
    pub fn new() -> Result<Self> {
        let ptr = unsafe { zt_schema_create() };
        if ptr.is_null() { Err(Error("schema create failed".into())) }
        else { Ok(Self { ptr, field_names: Vec::new() }) }
    }

    pub fn add_field(&mut self, name: &str, kind: Kind) -> Result<i32> {
        let cname = CString::new(name).map_err(|e| Error(e.to_string()))?;
        let mut err = errbuf();
        let idx = self.field_names.len() as i32;
        let rc = unsafe {
            zt_schema_add_field(self.ptr, cname.as_ptr(), kind as c_int,
                                1.0, err.as_mut_ptr(), err.len())
        };
        check(rc, &err)?;
        self.field_names.push(cname);
        Ok(idx)
    }

    pub fn add_choice(&mut self, field: i32, label: &str, tokens: &[i32], value: f32) -> Result<()> {
        let clabel = CString::new(label).map_err(|e| Error(e.to_string()))?;
        let mut err = errbuf();
        let rc = unsafe {
            zt_schema_add_choice(self.ptr, field, clabel.as_ptr(),
                                  tokens.as_ptr(), tokens.len() as u32, value,
                                  err.as_mut_ptr(), err.len())
        };
        check(rc, &err)
    }

    pub fn compile(self) -> Result<Program> {
        let mut err = errbuf();
        let ptr = unsafe { zt_schema_compile(self.ptr, err.as_mut_ptr(), err.len()) };
        if ptr.is_null() {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            Err(Error(msg))
        } else {
            Ok(Program { ptr, field_names: self.field_names })
        }
    }
}

impl Drop for Schema {
    fn drop(&mut self) { if !self.ptr.is_null() { unsafe { zt_schema_destroy(self.ptr) } } }
}

// ------------------------------------------------------------------ Program

pub struct Program { ptr: *mut c_void, pub field_names: Vec<CString> }

impl Program {
    pub fn n_fields(&self) -> u32 { unsafe { zt_program_n_fields(self.ptr) } }
}

impl Drop for Program {
    fn drop(&mut self) { if !self.ptr.is_null() { unsafe { zt_program_destroy(self.ptr) } } }
}

unsafe impl Send for Program {}

// ------------------------------------------------------------------ Engine

pub struct Engine { ptr: *mut c_void }

impl Engine {
    pub fn new(backend: Backend) -> Result<Self> {
        let mut err = errbuf();
        let ptr = unsafe { zt_engine_create(backend as c_int, err.as_mut_ptr(), err.len()) };
        if ptr.is_null() {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            Err(Error(msg))
        } else { Ok(Self { ptr }) }
    }

    pub fn session<'a>(&'a self, prog: &'a Program, max_batch: u32) -> Result<Session<'a>> {
        let mut err = errbuf();
        let ptr = unsafe {
            zt_session_create(self.ptr, prog.ptr, max_batch, err.as_mut_ptr(), err.len())
        };
        if ptr.is_null() {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            Err(Error(msg))
        } else { Ok(Session { ptr, prog }) }
    }
}

impl Drop for Engine {
    fn drop(&mut self) { if !self.ptr.is_null() { unsafe { zt_engine_destroy(self.ptr) } } }
}

unsafe impl Send for Engine {}

// ------------------------------------------------------------------ Session

pub struct Session<'a> { ptr: *mut c_void, prog: &'a Program }

impl<'a> Session<'a> {
    pub fn bind_head(&mut self, data: &[f32], rows: u32, cols: u32) -> Result<()> {
        let mut err = errbuf();
        let rc = unsafe {
            zt_session_bind_head(self.ptr, data.as_ptr() as *const c_void,
                                  rows, cols, err.as_mut_ptr(), err.len())
        };
        check(rc, &err)
    }

    pub fn run_logits(&mut self, logits: &[f32], vocab_size: u32, out: &mut ResultBuf) -> Result<()> {
        let rows  = logits.len() as u32 / vocab_size;
        let batch = rows / self.prog.n_fields();
        let mut err = errbuf();
        let rc = unsafe {
            zt_session_run(self.ptr, 0 /* LOGITS */,
                           logits.as_ptr() as *const c_void, vocab_size,
                           rows, batch,
                           out.dec.as_mut_ptr(), std::ptr::null_mut(),
                           err.as_mut_ptr(), err.len())
        };
        out.batch = batch;
        check(rc, &err)
    }
}

impl<'a> Drop for Session<'a> {
    fn drop(&mut self) { if !self.ptr.is_null() { unsafe { zt_session_destroy(self.ptr) } } }
}

// ------------------------------------------------------------------ ResultBuf

pub struct ResultBuf { pub dec: Vec<ZtDecision>, pub batch: u32 }

impl ResultBuf {
    pub fn new(prog: &Program, max_batch: u32, _with_probs: bool) -> Result<Self> {
        let cap = (max_batch * prog.n_fields()) as usize;
        let mut dec = Vec::with_capacity(cap);
        for _ in 0..cap {
            dec.push(ZtDecision { class_index: 0, confidence: 0.0, bitmask: 0, valid: 0, _pad: [0; 3] });
        }
        Ok(Self { dec, batch: 0 })
    }

    pub fn json(&self, batch_idx: u32, _flags: u32) -> Result<String> {
        let d = &self.dec[batch_idx as usize];
        Ok(format!("{{\"class_index\":{},\"confidence\":{:.4},\"valid\":{}}}",
                   d.class_index, d.confidence, d.valid))
    }
}
