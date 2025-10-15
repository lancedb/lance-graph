use crate::error::{GraphError, Result};
use datafusion::logical_expr::JoinType;

// Internal helper that plans and executes a single path by chaining joins.
pub(super) struct PathExecutor<'a> {
    pub(super) ctx: &'a datafusion::prelude::SessionContext,
    pub(super) path: &'a crate::ast::PathPattern,
    pub(super) start_label: &'a str,
    pub(super) start_alias: String,
    segs: Vec<SegMeta<'a>>,
    node_maps: std::collections::HashMap<String, &'a crate::config::NodeMapping>,
    rel_maps: std::collections::HashMap<String, &'a crate::config::RelationshipMapping>,
}

#[derive(Clone)]
struct SegMeta<'a> {
    rel_type: &'a str,
    end_label: &'a str,
    dir: crate::ast::RelationshipDirection,
    rel_alias: String,
    end_alias: String,
}

impl<'a> PathExecutor<'a> {
    pub(super) fn new(
        ctx: &'a datafusion::prelude::SessionContext,
        cfg: &'a crate::config::GraphConfig,
        path: &'a crate::ast::PathPattern,
    ) -> Result<Self> {
        use std::collections::{HashMap, HashSet};
        let mut used: HashSet<String> = HashSet::new();
        let mut uniq = |desired: &str| -> String {
            if used.insert(desired.to_string()) {
                return desired.to_string();
            }
            let mut i = 2usize;
            loop {
                let cand = format!("{}_{}", desired, i);
                if used.insert(cand.clone()) {
                    break cand;
                }
                i += 1;
            }
        };

        let start_label = path
            .start_node
            .labels
            .first()
            .map(|s| s.as_str())
            .ok_or_else(|| GraphError::PlanError {
                message: "Start node must have a label".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let start_alias = uniq(
            &path
                .start_node
                .variable
                .as_deref()
                .unwrap_or(start_label)
                .to_lowercase(),
        );

        let mut segs: Vec<SegMeta> = Vec::with_capacity(path.segments.len());
        for seg in &path.segments {
            let rel_type = seg
                .relationship
                .types
                .first()
                .map(|s| s.as_str())
                .ok_or_else(|| GraphError::PlanError {
                    message: "Relationship must have a type".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            let end_label = seg
                .end_node
                .labels
                .first()
                .map(|s| s.as_str())
                .ok_or_else(|| GraphError::PlanError {
                    message: "End node must have a label".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            let rel_alias = uniq(
                &seg.relationship
                    .variable
                    .as_deref()
                    .unwrap_or(rel_type)
                    .to_lowercase(),
            );
            let end_alias = uniq(
                &seg.end_node
                    .variable
                    .as_deref()
                    .unwrap_or(end_label)
                    .to_lowercase(),
            );
            segs.push(SegMeta {
                rel_type,
                end_label,
                dir: seg.relationship.direction.clone(),
                rel_alias,
                end_alias,
            });
        }

        let mut node_maps: HashMap<String, &crate::config::NodeMapping> = HashMap::new();
        let mut rel_maps: HashMap<String, &crate::config::RelationshipMapping> = HashMap::new();
        node_maps.insert(
            start_alias.clone(),
            cfg.get_node_mapping(start_label)
                .ok_or_else(|| GraphError::PlanError {
                    message: format!("No node mapping for '{}'", start_label),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?,
        );
        for seg in &segs {
            node_maps.insert(
                seg.end_alias.clone(),
                cfg.get_node_mapping(seg.end_label)
                    .ok_or_else(|| GraphError::PlanError {
                        message: format!("No node mapping for '{}'", seg.end_label),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?,
            );
            rel_maps.insert(
                seg.rel_alias.clone(),
                cfg.get_relationship_mapping(seg.rel_type).ok_or_else(|| {
                    GraphError::PlanError {
                        message: format!("No relationship mapping for '{}'", seg.rel_type),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    }
                })?,
            );
        }

        Ok(Self {
            ctx,
            path,
            start_label,
            start_alias,
            segs,
            node_maps,
            rel_maps,
        })
    }

    async fn open_aliased(
        &self,
        table: &str,
        alias: &str,
    ) -> Result<datafusion::dataframe::DataFrame> {
        let df = self
            .ctx
            .table(table)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to read table '{}': {}", table, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let schema = df.schema();
        let proj: Vec<datafusion::logical_expr::Expr> = schema
            .fields()
            .iter()
            .map(|f| {
                datafusion::logical_expr::col(f.name()).alias(format!("{}__{}", alias, f.name()))
            })
            .collect();
        df.alias(alias)?
            .select(proj)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to alias/select '{}': {}", table, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    pub(super) async fn build_chain(&self) -> Result<datafusion::dataframe::DataFrame> {
        // Start node
        let mut df = self
            .open_aliased(self.start_label, &self.start_alias)
            .await?;
        // Inline property filters on start node
        for (k, v) in &self.path.start_node.properties {
            let expr = super::expr::to_df_literal(v);
            df = df
                .filter(datafusion::logical_expr::Expr::BinaryExpr(
                    datafusion::logical_expr::BinaryExpr {
                        left: Box::new(datafusion::logical_expr::col(format!(
                            "{}__{}",
                            self.start_alias, k
                        ))),
                        op: datafusion::logical_expr::Operator::Eq,
                        right: Box::new(expr),
                    },
                ))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply filter: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        // Chain joins for each hop
        let mut current_node_alias = self.start_alias.as_str();
        for s in &self.segs {
            let rel_df = self.open_aliased(s.rel_type, &s.rel_alias).await?;
            let node_map = self.node_maps.get(current_node_alias).unwrap();
            let rel_map = self.rel_maps.get(&s.rel_alias).unwrap();
            let (left_key, right_key) = match s.dir {
                crate::ast::RelationshipDirection::Outgoing
                | crate::ast::RelationshipDirection::Undirected => (
                    format!("{}__{}", current_node_alias, node_map.id_field),
                    format!("{}__{}", s.rel_alias, rel_map.source_id_field),
                ),
                crate::ast::RelationshipDirection::Incoming => (
                    format!("{}__{}", current_node_alias, node_map.id_field),
                    format!("{}__{}", s.rel_alias, rel_map.target_id_field),
                ),
            };
            df = df
                .join(
                    rel_df,
                    JoinType::Inner,
                    &[left_key.as_str()],
                    &[right_key.as_str()],
                    None,
                )
                .map_err(|e| GraphError::PlanError {
                    message: format!("Join failed (node->rel): {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let end_df = self.open_aliased(s.end_label, &s.end_alias).await?;
            let (left_key2, right_key2) = match s.dir {
                crate::ast::RelationshipDirection::Outgoing
                | crate::ast::RelationshipDirection::Undirected => (
                    format!("{}__{}", s.rel_alias, rel_map.target_id_field),
                    format!(
                        "{}__{}",
                        s.end_alias,
                        self.node_maps.get(&s.end_alias).unwrap().id_field
                    ),
                ),
                crate::ast::RelationshipDirection::Incoming => (
                    format!("{}__{}", s.rel_alias, rel_map.source_id_field),
                    format!(
                        "{}__{}",
                        s.end_alias,
                        self.node_maps.get(&s.end_alias).unwrap().id_field
                    ),
                ),
            };
            df = df
                .join(
                    end_df,
                    JoinType::Inner,
                    &[left_key2.as_str()],
                    &[right_key2.as_str()],
                    None,
                )
                .map_err(|e| GraphError::PlanError {
                    message: format!("Join failed (rel->node): {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            current_node_alias = &s.end_alias;
        }

        Ok(df)
    }

    fn resolve_var_alias<'b>(&'b self, var: &str) -> Option<&'b str> {
        if Some(var) == self.path.start_node.variable.as_deref() {
            return Some(self.start_alias.as_str());
        }
        for (i, seg) in self.path.segments.iter().enumerate() {
            if Some(var) == seg.relationship.variable.as_deref() {
                return Some(self.segs[i].rel_alias.as_str());
            }
            if Some(var) == seg.end_node.variable.as_deref() {
                return Some(self.segs[i].end_alias.as_str());
            }
        }
        None
    }

    pub(super) fn apply_where(
        &self,
        df: datafusion::dataframe::DataFrame,
        ast: &crate::ast::CypherQuery,
    ) -> Result<datafusion::dataframe::DataFrame> {
        super::clauses::apply_where_with_qualifier(df, ast, &|var, prop| {
            let alias = self.resolve_var_alias(var).unwrap_or(var);
            super::aliases::qualify_alias_property(alias, prop)
        })
    }

    pub(super) fn apply_return(
        &self,
        df: datafusion::dataframe::DataFrame,
        ast: &crate::ast::CypherQuery,
    ) -> Result<datafusion::dataframe::DataFrame> {
        super::clauses::apply_return_with_qualifier(df, ast, &|var, prop| {
            let alias = self.resolve_var_alias(var).unwrap_or(var);
            super::aliases::qualify_alias_property(alias, prop)
        })
    }
}
