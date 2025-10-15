Contributing (Rust crate)
=========================

Before pushing or opening a PR, please verify Rust code formatting matches CI:

```bash
cargo fmt --manifest-path rust/lance-graph/Cargo.toml -- --check
```

If the check fails, format the code locally:

```bash
cargo fmt --manifest-path rust/lance-graph/Cargo.toml
```
