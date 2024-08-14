module ComplexityModule

using DynamicExpressions: AbstractScalarExprNode, count_nodes, tree_mapreduce
using ..CoreModule: Options, ComplexityMapping

function past_complexity_limit(tree::AbstractScalarExprNode, options::Options, limit)::Bool
    return compute_complexity(tree, options) > limit
end

"""
Compute the complexity of a tree.

By default, this is the number of nodes in a tree.
However, it could use the custom settings in options.complexity_mapping
if these are defined.
"""
function compute_complexity(
    tree::AbstractScalarExprNode, options::Options, operators::AbstractOperatorEnum; break_sharing=Val(false)
)::Int
    if options.complexity_mapping.use
        raw_complexity = _compute_complexity(
            tree, options.complexity_mapping; break_sharing
        )
        return round(Int, raw_complexity)
    else
        return count_nodes(tree; break_sharing)
    end
end

function compute_complexity(
    tree::AbstractTensorExprNode, options::Options, operators::AbstractTensorOperatorEnum; break_sharing=Val(false)
)::Int
    if options.complexity_mapping.use
        return tree_mapreduce(
            let vc = cmap.variable_complexity, cc = cmap.constant_complexity
                if vc isa AbstractVector
                    t -> t.constant ? cc*prod(t.shape) : @inbounds(vc[t.feature])*prod(t.shape)
                else
                    t -> t.constant ? cc*prod(t.shape) : vc*prod(t.shape)
                end
            end,
            let uc = cmap.unaop_complexities, bc = cmap.binop_complexities
                t -> if t.degree == 1
                    # TODO: maybe should use a @nif here
                    @inbounds(uc[t.op]) * @inbounds(operators.binops[t.op]).complexity(t.l.shape)
                else
                    @inbounds(bc[t.op]) * @inbounds(operators.unaops[t.op]).complexity(t.l.shape, t.r.shape)
                end
            end,
            +,
            tree,
            CT;
            break_sharing=break_sharing,
            f_on_shared=(result, is_shared) -> is_shared ? result : zero(CT),
        )
    else

    end
    
    if options.complexity_mapping.use
        raw_complexity = _compute_complexity(
            tree, options.complexity_mapping; break_sharing
        )
        return round(Int, raw_complexity)
    else
        return count_nodes(tree; break_sharing)
    end
end

function _compute_complexity(
    tree::AbstractScalarExprNode, cmap::ComplexityMapping{CT}; break_sharing=Val(false)
)::CT where {CT}
    return tree_mapreduce(
        let vc = cmap.variable_complexity, cc = cmap.constant_complexity
            if vc isa AbstractVector
                t -> t.constant ? cc : @inbounds(vc[t.feature])
            else
                t -> t.constant ? cc : vc
            end
        end,
        let uc = cmap.unaop_complexities, bc = cmap.binop_complexities
            t -> t.degree == 1 ? @inbounds(uc[t.op]) : @inbounds(bc[t.op])
        end,
        +,
        tree,
        CT;
        break_sharing=break_sharing,
        f_on_shared=(result, is_shared) -> is_shared ? result : zero(CT),
    )
end

end
