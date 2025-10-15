pub(super) fn to_df_boolean_expr_with_vars<F>(
    expr: &crate::ast::BooleanExpression,
    qualify: &F,
) -> Option<datafusion::logical_expr::Expr>
where
    F: Fn(&str, &str) -> String,
{
    use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO, ValueExpression as VE};
    use datafusion::logical_expr::{col, Expr, Operator};
    match expr {
        BE::Comparison {
            left,
            operator,
            right,
        } => {
            let (var, prop, lit_expr) = match (left, right) {
                (VE::Property(p), VE::Literal(val)) => {
                    (p.variable.as_str(), p.property.as_str(), to_df_literal(val))
                }
                (VE::Literal(val), VE::Property(p)) => {
                    (p.variable.as_str(), p.property.as_str(), to_df_literal(val))
                }
                _ => return None,
            };
            let qualified = qualify(var, prop);
            let op = match operator {
                CO::Equal => Operator::Eq,
                CO::NotEqual => Operator::NotEq,
                CO::LessThan => Operator::Lt,
                CO::LessThanOrEqual => Operator::LtEq,
                CO::GreaterThan => Operator::Gt,
                CO::GreaterThanOrEqual => Operator::GtEq,
            };
            Some(Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
                left: Box::new(col(&qualified)),
                op,
                right: Box::new(lit_expr),
            }))
        }
        BE::And(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_with_vars(l, qualify)?),
                op: Operator::And,
                right: Box::new(to_df_boolean_expr_with_vars(r, qualify)?),
            },
        )),
        BE::Or(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_with_vars(l, qualify)?),
                op: Operator::Or,
                right: Box::new(to_df_boolean_expr_with_vars(r, qualify)?),
            },
        )),
        BE::Not(inner) => Some(datafusion::logical_expr::Expr::Not(Box::new(
            to_df_boolean_expr_with_vars(inner, qualify)?,
        ))),
        _ => None,
    }
}

pub(super) fn to_df_literal(val: &crate::ast::PropertyValue) -> datafusion::logical_expr::Expr {
    use datafusion::logical_expr::lit;
    match val {
        crate::ast::PropertyValue::String(s) => lit(s.clone()),
        crate::ast::PropertyValue::Integer(i) => lit(*i),
        crate::ast::PropertyValue::Float(f) => lit(*f),
        crate::ast::PropertyValue::Boolean(b) => lit(*b),
        crate::ast::PropertyValue::Null => {
            datafusion::logical_expr::Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
        }
        crate::ast::PropertyValue::Parameter(_) => lit(0),
        crate::ast::PropertyValue::Property(prop) => datafusion::logical_expr::col(&prop.property),
    }
}
