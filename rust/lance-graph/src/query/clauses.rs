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
    // ORDER BY
    if let Some(order_by) = &ast.order_by {
        use datafusion::logical_expr::SortExpr;
        let mut sorts: Vec<SortExpr> = Vec::new();
        for item in &order_by.items {
            if let crate::ast::ValueExpression::Property(prop) = &item.expression {
                let col_name = qualify(&prop.variable, &prop.property);
                let col = datafusion::logical_expr::col(col_name);
                let asc = matches!(item.direction, crate::ast::SortDirection::Ascending);
                sorts.push(SortExpr {
                    expr: col,
                    asc,
                    nulls_first: false,
                });
            }
        }
        if !sorts.is_empty() {
            df = df.sort(sorts).map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply ORDER BY: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }
    }
    // SKIP/OFFSET and LIMIT
    if ast.skip.is_some() || ast.limit.is_some() {
        let offset = ast.skip.unwrap_or(0) as usize;
        let fetch = ast.limit.map(|l| l as usize);
        df = df.limit(offset, fetch).map_err(|e| GraphError::PlanError {
            message: format!("Failed to apply SKIP/LIMIT: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
    }
    Ok(df)
}
