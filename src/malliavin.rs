//! Port of `greeks/malliavin.py`.
//!
//! The Python file is a thin re-export module
//! (`from .greeks import Greeks, MalliavinGreeks; __all__ = [...]`) with no
//! numeric core of its own. We mirror it by re-exporting the implementation
//! from [`crate::greeks`].

pub use crate::greeks::{Greeks, MalliavinGreeks};
