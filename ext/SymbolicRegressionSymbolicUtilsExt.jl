module SymbolicRegressionSymbolicUtilsExt

using SymbolicUtils: Symbolic
using SymbolicRegression: AbstractScalarExprNode, Node, Options
using SymbolicRegression.MLJInterfaceModule: AbstractSRRegressor, get_options

import SymbolicRegression: node_to_symbolic, symbolic_to_node

"""
    node_to_symbolic(tree::AbstractScalarExprNode, options::Options; kws...)

Convert an expression to SymbolicUtils.jl form.
"""
function node_to_symbolic(tree::AbstractScalarExprNode, options::Options; kws...)
    return node_to_symbolic(tree, options.operators; kws...)
end
function node_to_symbolic(tree::AbstractScalarExprNode, m::AbstractSRRegressor; kws...)
    return node_to_symbolic(tree, get_options(m); kws...)
end

"""
    symbolic_to_node(eqn::Symbolic, options::Options; kws...)

Convert a SymbolicUtils.jl expression to SymbolicRegression.jl's `Node` type.
"""
function symbolic_to_node(eqn::Symbolic, options::Options; kws...)
    return symbolic_to_node(eqn, options.operators; kws...)
end
function symbolic_to_node(eqn::Symbolic, m::AbstractSRRegressor; kws...)
    return symbolic_to_node(eqn, get_options(m); kws...)
end

function Base.convert(
    ::Type{Symbolic}, tree::AbstractScalarExprNode, options::Options; kws...
)
    return convert(Symbolic, tree, options.operators; kws...)
end
function Base.convert(
    ::Type{Symbolic}, tree::AbstractScalarExprNode, m::AbstractSRRegressor; kws...
)
    return convert(Symbolic, tree, get_options(m); kws...)
end

function Base.convert(
    ::Type{N}, x::Union{Number,Symbolic}, options::Options; kws...
) where {N<:AbstractScalarExprNode}
    return convert(N, x, options.operators; kws...)
end
function Base.convert(
    ::Type{N}, x::Union{Number,Symbolic}, m::AbstractSRRegressor; kws...
) where {N<:AbstractScalarExprNode}
    return convert(N, x, get_options(m); kws...)
end

end
