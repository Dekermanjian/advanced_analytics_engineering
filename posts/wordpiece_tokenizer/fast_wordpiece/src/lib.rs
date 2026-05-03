pub mod tokenizer;
pub mod pipeline;
pub mod training;
pub mod python;

use pyo3::prelude::*;
use python::bindings::PyTokenizer;

#[pymodule]
fn fast_wordpiece(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
