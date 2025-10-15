pub(super) fn qualify_alias_property(alias: &str, property: &str) -> String {
    format!("{}__{}", alias, property)
}
