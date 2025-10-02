use std::sync::LazyLock;

use pyo3::prelude::*;

mod executor;
mod graph;

pub(crate) static RT: LazyLock<executor::BackgroundExecutor> =
    LazyLock::new(executor::BackgroundExecutor::new);

#[pymodule]
fn _internal(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    graph::register_graph_module(py, m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
