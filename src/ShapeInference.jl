using DynamicExpressions: Node, AbstractNode, @extend_operators, OperatorEnum

c1 = Node(Array{Float64,4}; val=fill(1, (1, 1, 1, 1)))
c2 = Node(Array{Float64,4}; val=fill(2, (1, 1, 1, 1)))
c3 = Node(Array{Float64,4}; val=fill(3, (1, 1, 1, 1)))
c4 = Node(Array{Float64,4}; val=fill(4, (1, 1, 1, 1)))
c5 = Node(Array{Float64,4}; val=fill(4, (1, 1, 1, 1)))
x1 = Node(Array{Float64,4}; feature=1)
x2 = Node(Array{Float64,4}; feature=2)
x3 = Node(Array{Float64,4}; feature=3)

# dummy operators

MAX_M::Int64 = 1
MAX_C::Int64 = 1

function cross(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:Number,N}
    @assert N >= 1
    @assert size(x) == size(y)
    @assert size(x, 1) == 3
    return rand(size(x))
end

function matmult(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:Number,N}
    @assert N >= 2
    @assert size(x)[3:end] == size(y)[3:end]
    @assert size(x, 2) == size(y, 1)
    return rand(size(x, 1), size(x, 2), size(x)[3:end]...)
end

macro define_matmult(firstAxis, secondAxis, N, T)
    @assert typeof(firstAxis) == Int64
    @assert typeof(secondAxis) == Int64
    @assert typeof(N) == Int64
    @assert firstAxis != secondAxis
    funcname = symbol("mm" * string(firstAxis) * string(secondAxis))
    if secondAxis < firstAxis
        firstAxis, secondAxis = secondAxis, firstAxis
    end
    return quote
        function $(funcname)(x::AbstractArray{T,N}, y::AbstractArray{T,N})
            @assert size(x, $(secondAxis)) == size(y, $(firstAxis))
            @assert size(x)[1:($(firstAxis - 1))] == size(y)[1:($(firstAxis - 1))]
            @assert size(x)[($(firstAxis + 1)):($(secondAxis - 1))] ==
                size(y)[($(firstAxis + 1)):($(secondAxis - 1))]
            @assert size(x)[($(secondAxis + 1)):end] == size(y)[($(secondAxis + 1)):end]
            return dims = (
                size(x)[1:($(secondAxis - 1))]...,
                size(y, $(secondAxis)),
                size(x)[($(secondAxis + 1)):N]...,
            )
        end
    end
end

function transp(x::AbstractArray{T,N}) where {T<:Number,N}
    @assert N >= 2
    return rand(size(x, 2), size(x, 1), size(x)[3:end]...)
end

plus(x, y) = x .+ y
times(x, y) = x .* y

operators = OperatorEnum(;
    binary_operators=[plus, times, cross, matmult], unary_operators=[transp]
)
@extend_operators(operators, on_type = Array{Float64,4})

trees = [
    matmult(transp(c4), cross(plus(c1, times(x1, c2)), times(x2, c3))),
    matmult(matmult(c4, x3), times(plus(c1, times(x1, c2)), times(x2, c3))),
    matmult(c1, plus(c2, x2)),
    matmult(c3, plus(c4, matmult(transp(matmult(c1, x1)), matmult(c2, plus(c5, transp(x2))))))
]

# Suppose we have a tree with return type Array{BT, N}, where BT = underlying number type, N = maximum allowed dimensions
# If at any point we have a lower dimension count than N, we set the size in the extra dimensions to 1
# Now we want to obtain the shapes of each node in the tree

# We traverse the tree and give each node a number
# We make a list of all the shape variables of the nodes, labeled a₁, a₂, ...
# When we encounter an operation, we impose a constraint on the shape variables
# For example, the mathematical expression of the constraint for the transpose 
#   nodes:      t2 = transpose(t1)
#   notation:   shape(t1) == (a1,a2), shape(t2) == (a3,a4)
#   constraint: (a1,a2,a3,a4) ∈ {(n,m,m,n) | n,m ∈ NN}
# More examples:
#   For the input we can impose a constraint of the type
#     (a₂₅,a₂₆,a₂₇,a₂₈) ∈ {(3,1,1,1)}
#   For a broadcasted sum (given by Julia Array rules)
#     (a₂₄,a₃₂,a₂₀) ∈ {(n,n,n)} ∪ {(n,1,n)} ∪ {(n,n,1)}
#   For a matrix multiplication:
#     (a₄₁,a₄₂,a₁,a₂,a₃₇,a₃₈) ∈ {(n,m,n,p,p,m)}

# Struct defintions

mutable struct ConstraintElem
    # if `isConstant` is set to true, the element will be a costnant of value `value`
    # otherwise, it will be a variable in the set, with the `value` being the index of the variable
    isConstant::Bool
    value::UInt64
end

Base.zero(::Type{ConstraintElem}) = ConstraintElem(true, 0)
@inline function Base.zero(::Type{NTuple{M,ConstraintElem}}) where {M}
    return ntuple(Returns(zero(ConstraintElem)), Val(M))
end
@inline function Base.zero(::Type{NTuple{C,NTuple{M,ConstraintElem}}}) where {M,C}
    return ntuple(Returns(zero(NTuple{M,ConstraintElem})), Val(C))
end

struct Constraint{M,C}
    # the shape variables (aᵢ)
    indices::NTuple{M,UInt64}
    # Union of sets of tuples of constant/variable elements, which determine all possible allowed valuas of the shape variable tuple
    consSets::NTuple{C,NTuple{M,ConstraintElem}}
end

@inline function Base.convert(
    ::Type{Constraint{M2,C2}}, x::Constraint{M1,C1}
) where {M1,M2,C1,C2}
    return Constraint{M2,C2}(
        ntuple(i -> i > M1 ? 0 : x.indices[i], Val(M2)),
        ntuple(
            i -> if i > C1
                zero(NTuple{M2,ConstraintElem})
            else
                ntuple(j -> j > M1 ? zero(ConstraintElem) : x.consSets[i][j], Val(M2))
            end,
            C2,
        ),
    )
end

# Helper functions

# with tuples
@inline function removeNth(a::NTuple{N,T}, ix::UInt64)::NTuple{N,T} where {N,T}
    return setNth(a, ix, zero(T))
end
@inline function setNth(a::NTuple{N,T}, ix::UInt64, val::T)::NTuple{N,T} where {N,T}
    return ntuple(i -> i == ix ? val : a[i], Val(N))
end
@inline function removeNth(a::Constraint{M,C}, ix::Int64)::Constraint{M,C} where {M,C}
    return Constraint{M,C}(
        removeNth(a.indices, ix),
        map(set -> reset_max_var(removeNth(set, a_x_index)), r.consSets),
    )
end

# with ConstraintElem
@inline Base.:(==)(a::ConstraintElem, b::ConstraintElem) =
    a.isConstant == b.isConstant && a.value == b.value

# with type-stable Constraint
@inline effective_M(a::Constraint{M,C}) where {M,C} = count(!=(0), a.indices)
@inline is_nonempty_set(set) = any(ce -> ce.value != 0, set)
@inline effective_C(a::Constraint{M,C}) where {M,C} = count(is_nonempty_set, a.consSets)

@inline same_indices(a::Constraint{M,C}, b::Constraint{M,C}) where {M,C} = 
    effective_M(a) == effective_M(b) && all(x -> x in a.indices, b.indices) 

function reset_max_var(r::NTuple{M,ConstraintElem})::NTuple{M,ConstraintElem} where {M}
    covered = count(x -> x.isConstant, r)
    i = 1
    while covered < M
        mins = minimum(map(x -> (x.isConstant || x.value < i) ? M + 5 : x.value, r))
        new_count = count(x -> !x.isConstant && x.value == mins, r)
        covered += new_count
        if mins != i
            r = map(
                x -> (x.isConstant || x.value != mins) ? x : ConstraintElem(false, i), r
            )
        end
        i += 1
    end
    return r
end

function resolve_set_value(
    r::Constraint{M,C}, a_x::UInt64, value::UInt64
)::Constraint{M,C} where {M,C}
    !(a_x in r.indices) && return r
    a_x_index = UInt64(findfirst(x -> x == a_x, r.indices))
    is_single = count(x -> x != 0, r.indices) == 1
    if is_single
        has_the_value = any(set -> set[a_x_index].isConstant && set[a_x_index].value == value, r.consSets)
        if !has_the_value
            return convert(Constraint{M,C}, Constraint{1,0}((a_x,), Tuple{}()))
        end
    end
    @show value
    return Constraint(
        removeNth(r.indices, a_x_index),
        map(
            set -> if set[a_x_index].isConstant
                set[a_x_index] == 0 ? set : removeNth(set, a_x_index)
            else
                varval = set[a_x_index].value
                x = set
                @show x
                x = map(
                    ce -> if !ce.isConstant && ce.value == varval
                        ConstraintElem(true, value)
                    else
                        ce
                    end,
                    set
                )
                @show x
                reset_max_var(
                    removeNth(
                        map(
                            ce -> if !ce.isConstant && ce.value == varval
                                ConstraintElem(true, value)
                            else
                                ce
                            end,
                            set,
                        ),
                        a_x_index,
                    ),
                )
            end,
            map(
                set -> if (set[a_x_index].isConstant && set[a_x_index].value != value)
                    zero(typeof(set))
                else
                    set
                end,
                r.consSets,
            ),
        ),
    )
end

# is A a subset of B ?
function Base.issubset(
    a::NTuple{M,ConstraintElem}, b::NTuple{M,ConstraintElem}
)::Bool where {M}
    constsMap = zero(NTuple{M,ConstraintElem})
    for i in eachindex(a)
        if b[i].isConstant
            if (a[i].isConstant && b[i].value != a[i].value) || !a[i].isConstant
                return false
            end
        else
            if constsMap[b[i].value].value == 0
                # if we don't know the value of 
                constsMap = setNth(constsMap, b[i].value, a[i])
            elseif !(constsMap[b[i].value] == a[i])
                return false
            end
        end
    end
    return true
end

function combine_constraint(
    a::Constraint{M,C}, b::Constraint{M,C}
)::Constraint{M,C} where {M,C}
    c2 = Constraint{M,2*C}(a.indices, (a.consSets..., b.consSets...))
    @assert effective_C(c2) <= C
    c = convert(Constraint{M,C}, make_canon(c2))
end

# Printing constraints
using Unicode
function to_subscript_str(n)
    return map(
        k -> Unicode.julia_chartransform(
            Unicode.julia_chartransform('₁') + Int(k) - Int('1')
        ),
        string(n),
    )
end
function parensd(va, func, io)
    if length(va) == 1
        print(io, func(va[1]))
    else
        print(io, "(")
        for i in eachindex(va)
            print(io, func(va[i]))
            if i != length(va)
                print(io, ",")
            end
        end
        print(io, ")")
    end
end

function Base.show(io::IO, ce::ConstraintElem)
    if ce.isConstant
        print(io, ce.value)
    else
        print(io, "_nmpqrtuvxyzbcdefghijka"[ce.value + 1])
    end
end

function Base.show(io::IO, c::Constraint)
    parensd(filter(x -> x != 0, c.indices), i -> "a" * to_subscript_str(i), io)
    print(io, " ∈ ")
    printed = false
    for i in eachindex(c.consSets)
        c.consSets[i] == zero(eltype(c.consSets)) && continue
        if printed
            print(io, " ∪ ")
        end
        print(io, "{")
        parensd(filter(ce -> ce.value != 0, c.consSets[i]), ce -> ce, io)
        print(io, "}")
        printed = true
    end
    if !printed
        print(io, "∅")
    end
end

function print_constraints(cs::Vector{Constraint{C,M}}) where {C,M}
    println("------- CONSTRAINTS --------")
    for i in eachindex(cs)
        count(!=(0), cs[i].indices) == 0 && continue
        println("R", i, ": ", cs[i])
    end
    println()
end

function flatten_tree!(
    v::Vector{Node{AT}}, tree::Node{AT}
) where {BT,N,AT<:AbstractArray{BT,N}}
    if tree.degree == 0
        push!(v, tree)
    elseif tree.degree == 1
        flatten_tree!(v, tree.l)
        push!(v, tree)
    elseif tree.degree == 2
        flatten_tree!(v, tree.l)
        flatten_tree!(v, tree.r)
        push!(v, tree)
    end
end

# A push constraints macro to make constraints a lot more readable
macro push_constraint(cs, indices, sets...)
    (length(sets) == 0 || indices.head != :tuple) && throw("Wrong format of indices")
    N = length(indices.args)
    C = length(sets)
    global MAX_M
    MAX_M = max(N, MAX_M)
    global MAX_C
    MAX_C = max(C, MAX_C)

    for i in eachindex(sets)
        (
            sets[i].head != :tuple ||
            length(sets[i].args) != N ||
            !all(x -> typeof(x) == Int64 ? x >= 1 : typeof(x) == Symbol, sets[i].args)
        ) && throw("Wrong format of $(i)th set")
        d = Dict{Symbol,Int64}()
        for j in eachindex(sets[i].args)
            if typeof(sets[i].args[j]) == Int64
                sets[i].args[j] = ConstraintElem(true, sets[i].args[j])
            elseif haskey(d, sets[i].args[j])
                sets[i].args[j] = ConstraintElem(false, d[sets[i].args[j]])
            else
                d[sets[i].args[j]] = length(d) + 1
                sets[i].args[j] = ConstraintElem(false, d[sets[i].args[j]])
            end
        end
    end

    return quote
        push!(
            $(esc(cs)),
            convert(
                eltype($(esc(cs))),
                Constraint{$(N),$(C)}(
                    $(esc(indices)),
                    $(NTuple{C,NTuple{N,ConstraintElem}}(
                        map(set -> NTuple{N,ConstraintElem}(set.args), sets)
                    )),
                ),
            ),
        )
    end
end

function push_constraint_costants(
    cs::AbstractVector{Constraint{M,C}}, start_index, constants::NTuple{N,Int64}
) where {N,M,C}
    return push!(
        cs,
        convert(
            Constraint{M,C},
            Constraint{N,1}(
                ntuple(x -> (start_index - 1) * N + x, Val(N)),
                (map(x -> ConstraintElem(true, x), constants),),
            ),
        ),
    )
end

# Function that appends the constraints determined by the operator f onto the constraint vector cs
# N - max number of tensor dimensions
# indices - indices of the dimension variables involved in the operations
#   indices[1]+1, indices[1]+2, ..., indices[1]+N represent the dimensions of the output node
#   indices[2]+1, indices[2]+2, ..., indices[2]+N represent the dimensions of the left node
#   indices[3]+1, indices[3]+2, ..., indices[3]+N represent the dimensions of the right node (if applicable)
# Unary operators have 2-tuple, while binary ops have a 3-tuple as argument into the function
function append_constraints!(
    f::F, cs::Vector{Constraint}, N::Int64, indices::Tuple{Int64,Int64}
) where {F}
    throw("Unimplemented")
end
function append_constraints!(
    f::F, cs::Vector{Constraint}, N::Int64, indices::Tuple{Int64,Int64,Int64}
) where {F}
    throw("Unimplemented")
end

# Implementations of append_constraints!

function append_constraints!(
    ::typeof(plus),
    cs2::Vector{Constraint{M,C}},
    N::Int64,
    indices::Tuple{Int64,Int64,Int64},
) where {M,C}
    p, l, r = indices
    for i in 1:N
        @push_constraint(cs2, (p + i, l + i, r + i), (n, n, n), (n, 1, n), (n, n, 1))
    end
end

function append_constraints!(
    ::typeof(times),
    cs1::Vector{Constraint{M,C}},
    N::Int64,
    indices::Tuple{Int64,Int64,Int64},
) where {M,C}
    p, l, r = indices
    for i in 1:N
        @push_constraint(cs1, (p + i, l + i, r + i), (n, n, n), (n, 1, n), (n, n, 1))
    end
end

function append_constraints!(
    ::typeof(transp), cs::Vector{Constraint{M,C}}, N::Int64, indices::Tuple{Int64,Int64}
) where {M,C}
    p, c = indices
    @push_constraint(cs, (p + 1, p + 2, c + 1, c + 2), (n, m, m, n))
    for i in 3:N
        @push_constraint(cs, (p + i, c + i), (n, n))
    end
end

function append_constraints!(
    ::typeof(matmult),
    cs::Vector{Constraint{M,C}},
    N::Int64,
    indices::Tuple{Int64,Int64,Int64},
) where {M,C}
    p, l, r = indices
    @push_constraint(cs, (p + 1, p + 2, l + 1, l + 2, r + 1, r + 2), (n, m, n, p, p, m))
    for i in 3:N
        @push_constraint(cs, (p + i, l + i, r + i), (n, n, n))
    end
end

function append_constraints!(
    ::typeof(cross),
    cs::Vector{Constraint{M,C}},
    N::Int64,
    indices::Tuple{Int64,Int64,Int64},
) where {M,C}
    p, l, r = indices
    @push_constraint(cs, (p + 1, l + 1, r + 1), (3, 3, 3))
    for i in 2:N
        @push_constraint(cs, (p + i, l + i, r + i), (n, n, n))
    end
end

function substitution(
    c::Constraint{M,C},
    cs::Vector{Constraint{M,C}},
    nodes::Vector{Node{AT}},
    solved::Dict{Node{AT},NTuple{N,Int64}},
)::Bool where {M,C,AT,N}
    effM, effC = effective_M(c), effective_C(c)
    !(effM == 1 && effC == 1) && return false
    a_x_index = findfirst(!=(0), c.indices)
    set_index = findfirst(s -> s[a_x_index].value != 0, c.consSets)
    !(c.consSets[set_index][a_x_index].isConstant) && return false
    @show a_x_index
    @show set_index
    a_x = c.indices[a_x_index]
    value = c.consSets[set_index][a_x_index].value
    node = nodes[div(a_x - 1, N) + 1]
    println("Applying substitution ", c)
    if solved[node][mod(a_x-1,N)+1] != 0
        # nooo
        print(map(node -> solved[node], nodes))
        error("Already set " * string(solved[node]))
    end
    map!(cst -> resolve_set_value(cst, a_x, value), cs, cs)
    print_constraints(cs)
    solved[node] = setNth(solved[node], UInt64(mod(a_x-1,N)+1), Int64(value))
    return true
end

function splitting(
    c::Constraint{M,C}, cs::Vector{Constraint{M,C}}, ci::Int64
)::Bool where {M,C}
    effM, effC = effective_M(c), effective_C(c)
    (effM == 0 || effC == 0 || effM == 1) && return false
    first = findfirst(set -> any(ce -> ce.value != 0, set), c.consSets)
    mask::NTuple{M,UInt64} = map(ce -> ce.isConstant ? ce.value : 0, c.consSets[first])
    for si in (first + 1):C
        for cei in 1:M
            !any(ce -> ce.value != 0, c.consSets[si]) && continue
            if (
                !c.consSets[si][cei].isConstant || mask[cei] != c.consSets[si][cei].value
            ) && mask[cei] != 0
                mask = setNth(mask, UInt64(cei), UInt64(0))
            end
        end
    end
    sum(mask) == 0 && return false
    println("Applying splitting for ", c)
    for ix in 1:M
        if mask[ix] != 0
            push!(
                cs,
                convert(
                    Constraint{M,C},
                    Constraint{1,1}((c.indices[ix],), ((ConstraintElem(true, mask[ix]),),)),
                ),
            )
        end
    end
    cs[ci] = Constraint{M,C}(
        ntuple(i -> mask[i] != 0 ? 0 : c.indices[i], Val(M)),
        map(
            set -> ntuple(i -> mask[i] != 0 ? zero(ConstraintElem) : set[i], Val(M)),
            c.consSets,
        ),
    )
    print_constraints(cs)
    return true
end

function simplification(
    c::Constraint{M,C}, cs::Vector{Constraint{M,C}}, ci::Int64
)::Bool where {M,C}
    effM, effC = effective_M(c), effective_C(c)
    (effM == 0 || effC == 1 || effC == 0) && return false
    mask = ntuple(
        si -> begin
            for pi in eachindex(c.consSets)
                pi == si && continue
                if issubset(c.consSets[si], c.consSets[pi]) && !(c.consSets[si] == c.consSets[pi])
                    return 1 # delete
                end
            end
            return 0 # keep
        end, Val(C)
    )
    sum(mask) == 0 && return false
    println("Applying simplification on ", cs[ci])
    cs[ci] = Constraint{M,C}(
        c.indices,
        ntuple(i -> mask[i] == 0 ? c.consSets[i] : zero(eltype(c.consSets)), Val(C)),
    )
    print_constraints(cs)
    return true
end

# Infer the tensor shapes of each node in the tree
# Given: the root node, the operator enum and the featureSizes, which represent the sizes of each feature (including the output, at the end)
# Will return the remaining constraints after everything is determined (there might still be indeterminancies)
#   and a dictionoary between nodes and their shapes (with 0 in place of indeterminancies)
#   or false if the expression is not correct
function _shape_inference(
    nodes::Vector{Node{AT}},
    operators::O,
    ::Val{M},
    ::Val{C},
    featureSizes::NTuple{K,NTuple{N,Int64}}
) where {M,C,BT,N,K,O<:OperatorEnum,AT<:AbstractArray{BT,N}}
    dict = Dict{Node{AT},Int64}()
    solved = Dict{Node{AT},NTuple{N,Int64}}()
    for i in eachindex(nodes)
        dict[nodes[i]] = i
        solved[nodes[i]] = ntuple(Returns(0), N)
    end

    println("------- FLATTENED TREE --------")
    for i in eachindex(nodes)
        parensd((i - 1) * N .+ collect(1:N), ix -> "a" * to_subscript_str(ix), stdout)
        println(" -> ", nodes[i])
    end
    println()

    # Adding all the constraints
    cs = Vector{Constraint{M,C}}()
    sizehint!(cs, length(nodes) * N)
    for i in eachindex(nodes)
        if i == length(nodes)
            push_constraint_costants(cs, i, featureSizes[end])
        end
        if nodes[i].degree == 0 && !nodes[i].constant
            push_constraint_costants(cs, i, featureSizes[nodes[i].feature])
        elseif nodes[i].degree == 1
            append_constraints!(
                operators.unaops[nodes[i].op],
                cs,
                N,
                ((i - 1) * N, (dict[nodes[i].l] - 1) * N),
            )
        elseif nodes[i].degree == 2
            append_constraints!(
                operators.binops[nodes[i].op],
                cs,
                N,
                ((i - 1) * N, (dict[nodes[i].l] - 1) * N, (dict[nodes[i].r] - 1) * N),
            )
        end
    end

    #print_constraints(cs)    

    # simplification loop
    should_continue = true
    while should_continue
        should_continue = false

        # Substitution
        for ci in eachindex(cs)
            should_continue |= substitution(cs[ci], cs, nodes, solved)
        end

        # Splitting
        for ci in eachindex(cs)
            should_continue |= splitting(cs[ci], cs, ci)
        end

        # Simplification
        for ci in eachindex(cs)
            should_continue |= simplification(cs[ci], cs, ci)
        end

        # Union
        for ci in eachindex(cs)
            for cj in 1:(ci-1)
                !same_indices(cs[ci], cs[cj]) && continue
                # cs[cj] = union()
                # cs[ci] = zero(Constraint{M,C})
            end
        end

        # Remove illegal or redundant constraints
        filter!(c -> count(!=(0), c.indices) != 0, cs)
        if count(c -> count(set -> any(ce -> ce.value != 0, set), c.consSets) == 0, cs) != 0
            indices = findall(
                ci -> count(set -> any(ce -> ce.value != 0, set), cs[ci].consSets) == 0,
                eachindex(cs),
            )
            for ix in indices
                # println(
                #     "Shapes of expressions ",
                #     map(i -> nodes[div(i - 1, N) + 1], cs[ix].indices),
                #     " do not match",
                # )
                # println(
                #     "Their determined shapes are: ",
                #     map(
                #         i -> solved[(div(i - 1, N) * N + 1):(div(i - 1, N) * N + N)],
                #         cs[ix].indices,
                #     ),
                # )
            end
            print(map(node -> solved[node], nodes))
            @warn "Expression cannot rigurously exist"
            return false
        end
    end

    println("Finished")
    print_constraints(cs)
    print(solved)
    return true
end

function shape_inference(
    tree::Node{AT}, operators::O, featureSizes::NTuple{K,NTuple{N,Int64}}
) where {BT,N,K,O<:OperatorEnum,AT<:AbstractArray{BT,N}}

    # Flatten tree and create useful dictionaries
    M = max(MAX_M, N)
    C = MAX_C

    nodes = Node{AT}[]
    flatten_tree!(nodes, tree)
    return _shape_inference(nodes, operators, Val(M), Val(C), featureSizes)
end
