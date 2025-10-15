use crate::error::Result;

pub(super) fn apply_where_with_qualifier(
    mut df: datafusion::dataframe::DataFrame,
    ast: &crate::ast::CypherQuery,
    qualify: &dyn Fn(&str, &str) -> String,
) -> Result<datafusion::dataframe::DataFrame> {
    use crate::error::GraphError;
    use crate::query::expr::to_df_boolean_expr_with_vars;
    if let Some(where_clause) = &ast.where_clause {
        if let Some(expr) =
            to_df_boolean_expr_with_vars(&where_clause.expression, &|v, p| qualify(v, p))
        {
            df = df.filter(expr).map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply WHERE: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }
    }
    Ok(df)
}

pub(super) fn apply_return_with_qualifier(
    mut df: datafusion::dataframe::DataFrame,
    ast: &crate::ast::CypherQuery,
    qualify: &dyn Fn(&str, &str) -> String,
) -> Result<datafusion::dataframe::DataFrame> {
    use crate::error::GraphError;
    use datafusion::logical_expr::Expr;
    let mut proj: Vec<Expr> = Vec::new();
    for item in &ast.return_clause.items {
        if let crate::ast::ValueExpression::Property(prop) = &item.expression {
            let col_name = qualify(&prop.variable, &prop.property);
            let mut e = datafusion::logical_expr::col(col_name);
            if let Some(a) = &item.alias {
                e = e.alias(a);
            }
            proj.push(e);
        }
    }
    if !proj.is_empty() {
        df = df.select(proj).map_err(|e| GraphError::PlanError {
            message: format!("Failed to project: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
    }
    if ast.return_clause.distinct {
        df = df.distinct().map_err(|e| GraphError::PlanError {
            message: format!("Failed to apply DISTINCT: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
    }
    if let Some(limit) = ast.limit {
        df = df
            .limit(0, Some(limit as usize))
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply LIMIT: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
    }
    Ok(df)
}
