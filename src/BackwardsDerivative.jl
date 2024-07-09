
using ..NodeModule: Node

# It's important to allocate the values from the begining as they will be reused a lot
mutable struct DifferentiableNodeValue{T}
    node::Node{T}
    value::T
    has_consts_l::Bool # whether a tree has constant children on its left side
    has_consts_r::Bool # whether a tree has constant children on its right side
    derivative_l::T
    derivative_r::T
end

mutable struct OptimizeFrame{T,BT,LFT,OPT}
    tree::T
    operators::OPT
    value::BT
    root_derivative::T
    loss_func::LFT
    constants::Vector{BT}
    derivatives::Vector{BT}
    refs::Vector{Ref{Node{T}}}
    index::Dict{Node{T}, DifferentiableNodeValue{T}}
    computed::UInt8
        # 0b001 -> updated the constants from the constants vector
        # 0b010 -> computed value
        # 0b100 -> computed derivative
end

function make_frame(tree::Node{AT}, cX::AbstractVector{ATP1}, shapes::Dict{Node{AT}, NTuple{N, Int64}}, operators::OperatorEnum, allocator::F, loss_function::LFT) where {BT,N,F,AT<:AbstractArray{BT,N},NP1,ATP1<:AbstractArray{BT,NP1},LFT} 
#= ::OptimizeFrame{AT,BT}, with F function from NTuple{N, Int64} to ATP1 or nothing to empty =#
    index = Dict{Node{T}, DifferentiableNodeValue{T}}()
    datapoints = size(cX[1], 1)
    add_node(node::Node{AT}) = begin
        if node.degree == 0
            index[node] = DifferentiableNodeValue(
                node, node.constant ? allocator(size(node.val)) : cX[node.feature], node.constant, false, node.constant ? allocator(shapes[node]) : allocator(), allocator()
            )
            return node.constant
        elseif node.degree == 1
            has_consts = add_node(node.l) 
            index[node] = DifferentiableNodeValue(
                node, allocator(shapes[node]), has_consts, false, has_consts ? allocator(shapes[node.l]) : allocator(), allocator()
            )
            return has_consts
        else
            has_consts_l = add_node(node.l)
            has_consts_r = add_node(node.r) 
            index[node] = DifferentiableNodeValue(
                node, allocator(shapes[node]), has_consts_l, has_consts_r, 
                has_consts_l ? allocator(shapes[node.l]) : allocator(), 
                has_consts_r ? allocator(shapes[node.r]) : allocator()
            )
            return has_consts_l || has_consts_r
        end
    end
    constants, refs = get_constants(tree)
    derivatives = Vector{BT}(undef, length(constants))
    root_derivative = allocator(shapes[tree])
    OptimizeFrame(tree, operators, zero(BT), root_derivative, loss_function, constants, derivatives, refs, index, 0)
end

function eval_frame_node(frame::OptimizeFrame{T,BT,LFT}, node::Node{T}) where {T, BT, LFT}
    
    if node.degree == 0
        if node.constant
            # hopefully there won't be an extra gpu allocation here
            index[node].value .= repeat(reshape(node.val, 1, size(node.val)...))
        end
    elseif node.degree == 1
        op = frame.operators.unaops[node.op]
        eval_frame_node(frame, node.l)
        index[node].value .= op(index[node.l].value)
    elseif node.degree == 2
        op = frame.operators.binops[node.op]
        eval_frame_node(frame, node.l)
        eval_frame_node(frame, node.r)
        index[node].value .= op(node.l, node.r)
    end

end

function eval_frame(frame::OptimizeFrame{T,BT,LFT}) where {T,BT,LFT}
    if frame.computed & 0b010 != 0
        return frame.value
    end
    if frame.computed & 0b001 == 0
        set_constants!(tree, constants, refs)
        frame.computed |= 0b001
    end
    eval_frame_node(frame, frame.tree)
    final_value = frame.index[free.tree].value
    frame.value = sum(i -> frame.loss_func(selectdim(final_value, 1, i)), axes(final_value, 1))

    frame.computed |= 0b010
    return frame.value
end

@inline multiply_jacobian(next_der, prev_der, J) = begin
    next_der .= reshape(transpose(J) * reshape(prev_der, length(prev_der)), size(next_der))
end

# TODO: it would be a lot, lot better to use these kinds of functions
# where we don't use a massive jacobian and we don't allocate anything
# downside: we have to define derivatives for everything, insted of it being done automatically

actually_good_derivative(::typeof(.+), ∂out, r, ∂r, l, ∂l) = begin
    sum!(∂r, ∂out)
    sum!(∂l, ∂out)
end

actually_good_derivative(::typeof(.-), ∂out, r, ∂r, l, ∂l) = begin
    sum!(∂l, ∂out)
    sum!(-, ∂r, ∂out)
end

actually_good_derivative(::typeof(.*), ∂out, r, ∂r, l, ∂l) = begin
    sum!(∂l, r .* ∂out)
    sum!(∂r, l .* ∂out)
end


function eval_diff_frame_node(frame::OptimizeFrame{T}, node::Node{T}, derivative::AT) where {T,AT}
    if node.degree == 0
        frame.index[node].derivative_l .= derivative
    elseif node.degree == 1
        op = frame.operators.unaops[node.op]
        inner_val = frame.index[node.l].value
        multiply_jacobian(frame.index[node].derivative_l, derivative, jacobian(op, inner_val)[1])
        eval_diff_frame_node(frame, node.l, frame.index[node].derivative_l)
    elseif node.degree == 2
        op = frame.operators.binops[node.op]
        op = frame.operators.unaops[node.op]
        val_l = frame.index[node.l].value
        val_r = frame.index[node.r].value
        # Jacobian
        (d_val_l, d_val_r) = jacobian(op, val_l, val_r)
        multiply_jacobian(frame.index[node].derivative_l, derivative, d_val_l)
        multiply_jacobian(frame.index[node].derivative_r, derivative, d_val_r)
        if frame.index[node.l].has_consts_l
            eval_diff_frame_node(frame, node.l, frame.index[node].derivative_l)
        end
        if frame.index[node.r].has_consts_r
            eval_diff_frame_node(frame, node.r, frame.index[node].derivative_r)
        end
    end
end

function set_derivatives(frame::OptimizeFrame{T}, node::Node{T}, idx, gradient_vector) where {T}
    if node.degree == 0
        if node.constant
            derivative = frame.index[node].derivative_l
            copyto!(@view(gradient_vector[idx:end]), reshape(sum(derivative; dims=1)./size(derivative,1), length(derivative)/size(derivative, 1)))
            idx += length(derivative)/size(derivative, 1)
        end
    elseif node.degree == 1
        idx = set_derivatives(frame, node.l, idx, gradient_vector)
    elseif node.degree == 2
        idx = set_derivatives(frame, node.l, idx, gradient_vector)
        idx = set_derivatives(frame, node.r, idx, gradient_vector)
    end
    return idx
end

function eval_diff_frame(frame::OptimizeFrame{T,BT,LFT}, gradient_vector=frame.derivatives) where {T,BT,LFT}
    if frame.computed & 0b100 != 0
        return frame.derivatives
    end
    eval_frame(frame)
    frame.root_gradient = gradient(
        final_value -> sum(i -> frame.loss_func(selectdim(final_value, 1, i)), axes(final_value, 1)),
        frame.index[tree].value
    )[1]
    eval_diff_frame_node(frame, frame.tree, frame.root_derivative)
    set_derivatives(frame, frame.tree, 0, gradient_vector)

    frame.computed |= 0b100
    return frame.derivatives
end

function shape_upgrade(shapes::Dict{Node{AT}, NTuple{N, Int64}}) where {BT,N,AT<:AbstractArray{BT,N}}
    for node in keys(shapes)
        (node.degree != 0 || !node.constant) && continue
        desired_shape = shapes[node]
        current_shape = size(node.val)
        if desired_shape != current_shape
            node.val = AT <: CUDA.CuArray ? CUDA.rand(desired_shape) : rand(desired_shape)
        end
    end
end

function optimize_constants!(tree::Node{AT}, operators::OperatorEnum, cX::AbstractVector{ATP1}, y::AbstractArray{ATP1}, loss_function) where {BT,N,AT<:AbstractArray{BT,N},NP1,ATP1<:AbstractArray{BT,N}}
    allocator = (sizes) -> error("Unkown array type")
    datasets = size(y, 1)
    @show AT
    return
    if AT <: Array{BT,N}
        allocator = (sizes) -> Array{BT,length(sizes)+1}(undef, datasets, sizes...)
    elseif AT <: CUDA.CuArray{BT,N}
        allocator = (sizes) -> CuArray{BT,length(sizes)+1}(undef, datasets, sizes...)
    end
    shapes = ntuple(i -> i > length(cX) ? size(y) : size(cX[i]), length(cX)+1)
    shapes_d, ok = shape_inference(tree, operators, shapes; throw_errors = false)
    !ok && error("Wrong shape!!!")
    shape_upgrade(shapes_d)
    frame = make_frame(tree, cX, shapes_d, operators, allocator, loss_function)
    optimize((new_constants) -> begin
        if new_constants != frame.constants
            frame.constants .= new_constants
            frame.computed = 0
        end
        return eval_frame(frame)
    end,
    (gradient_vector, new_constants) -> begin
        if new_constants != frame.constants
            frame.constants .= new_constants
            frame.computed = 0
        end
        return eval_diff_frame(frame, gradient_vector)
    end; inplace=true)
end