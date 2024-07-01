using DynamicExpressions: Node, AbstractNode, @extend_operators, OperatorEnum

c1 = Node(Array{Float64, 4}; val = fill(1, (1,1,1,1)))
c2 = Node(Array{Float64, 4}; val = fill(1, (1,1,1,1)))
c3 = Node(Array{Float64, 4}; val = fill(1, (1,1,1,1)))
c4 = Node(Array{Float64, 4}; val = fill(1, (1,1,1,1)))
x1 = Node(Array{Float64, 4}; feature=1)
x2 = Node(Array{Float64, 4}; feature=2)
x3 = Node(Array{Float64, 4}; feature=3)

# dummy operators

function cross(x::AbstractArray{T,N}, y::AbstractArray{T, N}) where {T<:Number, N}
    @assert N>=1
    @assert size(x) == size(y)
    @assert size(x, 1) == 3
    return rand(size(x))
end

function matmult(x::AbstractArray{T,N}, y::AbstractArray{T, N}) where {T<:Number, N}
    @assert N>=2
    @assert size(x)[3:end] == size(y)[3:end]
    @assert size(x, 2) == size(y, 1)
    return rand(size(x, 1), size(x, 2), size(x)[3:end]...)
end

function transp(x::AbstractArray{T, N}) where {T<:Number, N}
    @assert N>=2
    return rand(size(x, 2), size(x, 1), size(x)[3:end]...)
end

plus(x, y) = x .+ y
times(x, y) = x .* y

operators = OperatorEnum(; binary_operators=[plus, times, cross, matmult], unary_operators=[transp])
@extend_operators(operators, on_type=Array{Float64, 4})

tree1 = matmult(transp(c4), cross(plus(c1, times(x1, c2)), times(x2, c3)))
tree2 = matmult(matmult(c4, x3), times(plus(c1, times(x1, c2)), times(x2, c3)))
tree3 = matmult(c1, plus(c2, x2))

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

struct Constraint{N, C}
    # the shape variables (aᵢ)
    indices::NTuple{N, UInt64}
    # Union of sets of tuples of constant/variable elements, which determine all possible allowed valuas of the shape variable tuple
    consSets::NTuple{C, NTuple{N, ConstraintElem}}
end


# Helper functions

function removeNth(a::NTuple{N,T}, ix::Int64)::NTuple{N-1,T} where {N,T}
    return (a[1:(ix-1)]..., a[(ix+1):end]...)
end
function setNth(a::NTuple{N,T}, ix::UInt64, val::T)::NTuple{N,T} where {N,T}
    ntuple(i -> i == ix ? val : a[i], N)
end
function removeNth(a::Constraint{N,C}, ix::Int64) where {N,C}
    Constraint{N-1,C}(
        removeNth(a.indices, ix),
        map(set -> reset_max_var(removeNth(set, a_x_index)), r.consSets)
    )
end

function reset_max_var(r::NTuple{N,ConstraintElem})::NTuple{N,ConstraintElem} where {N}
    covered = count(x -> x.isConstant, r)
    i = 1
    while covered < N
        mins = minimum(map(x -> (x.isConstant || x.value < i) ? N+5 : x.value, r))
        new_count = count(x -> !x.isConstant && x.value == mins, r)
        covered += new_count
        if mins != i
            r = map(x -> (x.isConstant || x.value != mins) ? x : ConstraintElem(false, i), r) 
        end
        i += 1
    end
    return r
end

function resolve_set_value(r::Constraint{N,C}, a_x::UInt64, value::UInt64)::Constraint where {N,C}
    !(a_x in r.indices) && return r
    a_x_index = findfirst(x -> x == a_x, r.indices)
    
    Constraint(
        removeNth(r.indices, a_x_index),
        map(
            set -> if set[a_x_index].isConstant
                removeNth(set, a_x_index)
            else
                varval = set[a_x_index].value
                reset_max_var(removeNth(map(ce -> !ce.isConstant && ce.value == varval ? ConstraintElem(true, value) : ce, set), a_x_index))
            end, 
            filter(
                set -> !(set[a_x_index].isConstant && set[a_x_index].value != value), 
                r.consSets
            )
        )
    )
end

@inline Base.:(==)(a::ConstraintElem, b::ConstraintElem) = a.isConstant == b.isConstant && a.value == b.value

# is A a subset of B ?
function Base.issubset(a::NTuple{N, ConstraintElem}, b::NTuple{N, ConstraintElem})::Bool where {N}
    constsMap = ntuple(x -> ConstraintElem(true, 0), N)
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

# Printing constraints
using Unicode
to_subscript_str(n) = map(k -> Unicode.julia_chartransform(Unicode.julia_chartransform('₁')+Int(k)-Int('1')),string(n))
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
        print(io, "nmpqrtuvxyzbcdefghijka"[ce.value])
    end
end

function Base.show(io::IO, c::Constraint)
    parensd(c.indices, i -> "a" * to_subscript_str(i), io)
    print(io, " ∈ ")
    for i in eachindex(c.consSets)
        print(io, "{")
        parensd(c.consSets[i], ce -> ce, io)
        print(io, "}")
        if i != length(c.consSets)
            print(io, " ∪ ")
        end
    end
end

function flatten_tree!(v::Vector{Node{AT}}, tree::Node{AT}) where {BT,N,AT<:AbstractArray{BT, N}}
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
    
    for i in eachindex(sets)
        (sets[i].head != :tuple || length(sets[i].args) != N || 
            !all(x -> typeof(x) == Int64 ? x >= 1 : typeof(x) == Symbol, sets[i].args)) && 
            throw("Wrong format of $(i)th set")
        d = Dict{Symbol, Int64}()
        for j in eachindex(sets[i].args)
            if typeof(sets[i].args[j]) == Int64
                sets[i].args[j] = ConstraintElem(true, sets[i].args[j])
            elseif haskey(d, sets[i].args[j])
                sets[i].args[j] = ConstraintElem(false, d[sets[i].args[j]])
            else
                d[sets[i].args[j]] = length(d)+1
                sets[i].args[j] = ConstraintElem(false, d[sets[i].args[j]])
            end
        end
    end
 
    return quote push!(
        $(esc(cs)), Constraint{$(N), $(C)}(
            $(esc(indices)), $(NTuple{C,NTuple{N,ConstraintElem}}(map(set -> NTuple{N, ConstraintElem}(set.args), sets)))
        ))
    end

end

function push_constraint_costants(cs::AbstractVector{Constraint}, start_index, constants::NTuple{N, Int64}) where {N}
    push!(cs, Constraint{N,1}(
        ntuple(x -> (start_index-1)*N+x, N),
        (map(x -> ConstraintElem(true, x), constants),)
    ))
end

# Function that appends the constraints determined by the operator f onto the constraint vector cs
# N - max number of tensor dimensions
# indices - indices of the dimension variables involved in the operations
#   indices[1]+1, indices[1]+2, ..., indices[1]+N represent the dimensions of the output node
#   indices[2]+1, indices[2]+2, ..., indices[2]+N represent the dimensions of the left node
#   indices[3]+1, indices[3]+2, ..., indices[3]+N represent the dimensions of the right node (if applicable)
# Unary operators have 2-tuple, while binary ops have a 3-tuple as argument into the function
function append_constraints!(f::F, cs::Vector{Constraint}, N::Int64, indices::Tuple{Int64, Int64}) where {F} throw("Unimplemented") end
function append_constraints!(f::F, cs::Vector{Constraint}, N::Int64, indices::Tuple{Int64, Int64, Int64}) where {F} throw("Unimplemented") end

# Implementations of append_constraints!

function append_constraints!(::typeof(plus), cs2::Vector{Constraint}, N::Int64, indices::Tuple{Int64, Int64, Int64}) 
    p,l,r = indices
    for i in 1:N
        @push_constraint(cs2, (p+i, l+i, r+i), (n, n, n), (n, 1, n), (n, n, 1))
    end
end

function append_constraints!(::typeof(times), cs1::Vector{Constraint}, N::Int64, indices::Tuple{Int64, Int64, Int64}) 
    p,l,r = indices
    for i in 1:N
        @push_constraint(cs1, (p+i, l+i, r+i), (n, n, n), (n, 1, n), (n, n, 1))
    end
end

function append_constraints!(::typeof(transpose), cs::Vector{Constraint}, N::Int64, indices::Tuple{Int64, Int64}) 
    p,c = indices
    @push_constraint(cs, (p+1,p+2,c+1,c+2), (n,m,m,n))
    for i in 3:N
        @push_constraint(cs, (p+i, c+i), (n,n))
    end
end

function append_constraints!(::typeof(matmult), cs::Vector{Constraint}, N::Int64, indices::Tuple{Int64, Int64, Int64}) 
    p,l,r = indices
    @push_constraint(cs, (p+1, p+2, l+1, l+2, r+1, r+2), (n,m,n,p,p,m))
    for i in 3:N
        @push_constraint(cs, (p+i, l+i, r+i), (n,n,n))
    end
end

function append_constraints!(::typeof(cross), cs::Vector{Constraint}, N::Int64, indices::Tuple{Int64, Int64, Int64}) 
    p,l,r = indices
    @push_constraint(cs, (p+1,l+1,r+1), (3,3,3))
    for i in 2:N
        @push_constraint(cs, (p+i, l+i, r+i), (n,n,n))
    end
end

print_constraints(cs::Vector{Constraint}) = begin
    println("------- CONSTRAINTS --------")
    for i in eachindex(cs)
        length(cs[i].indices) == 0 && continue
        println("R", i, ": ", cs[i])
    end
    println()
end



# Infer the tensor shapes of each node in the tree
# Given: the root node, the operator enum and the featureSizes, which represent the sizes of each feature (including the output, at the end)
# Will return the remaining constraints after everything is determined (there might still be indeterminancies)
#   and a dictionoary between nodes and their shapes (with 0 in place of indeterminancies)
#   or false if the expression is not correct
function shape_inference(tree::Node{AT}, operators::O, featureSizes::NTuple{K,NTuple{N,Int64}}) where {BT, N, K, O<:OperatorEnum, AT<:AbstractArray{BT, N}}
    
    # Flatten tree and create useful dictionaries
    nodes = Node{AT}[]
    flatten_tree!(nodes, tree)
    dict = Dict{Node{AT}, Int64}()
    solved = Dict{Node{AT}, NTuple{N,Int64}}()
    for i in eachindex(nodes)
        dict[nodes[i]] = i
        solved[nodes[i]] = ntuple(Returns(0), N)
    end
    
    # println("------- FLATTENED TREE --------")
    # for i in eachindex(nodes)
    #     parensd((i-1)*N .+ collect(1:N), ix -> "a" * to_subscript_str(ix), stdout)
    #     println(" -> ", nodes[i])
    # end
    # println()
    
    # Adding all the constraints
    cs = Constraint[]
    for i in eachindex(nodes)
        if i == length(nodes)
            push_constraint_costants(cs, i, featureSizes[end])
        end
        if nodes[i].degree == 0 && !nodes[i].constant
            push_constraint_costants(cs, i, featureSizes[nodes[i].feature])
        elseif nodes[i].degree == 1
            append_constraints!(operators.unaops[nodes[i].op], cs, N, ((i-1)*N, (dict[nodes[i].l]-1)*N))
        elseif nodes[i].degree == 2
            append_constraints!(operators.binops[nodes[i].op], cs, N, ((i-1)*N, (dict[nodes[i].l]-1)*N, (dict[nodes[i].r]-1)*N))
        end
    end

    print_constraints(cs)    
    
    # simplification loop
    should_stop = false
    while !should_stop
        should_stop = true

        # Substitution
        for c in cs
            M = length(c.indices)
            C = length(c.consSets)
            !(M == 1 && C == 1 && c.consSets[1][1].isConstant) && continue
                
            #println("Applying substitution ", c)
            a_x = c.indices[1]
            value = c.consSets[1][1].value
            map!(cst -> resolve_set_value(cst, a_x, value), cs, cs)
            should_stop = false
            node = nodes[div(a_x-1, N)+1]
            solved[node] = setNth(solved[node], a_x, Int64(value))

            #print_constraints(cs)    

        end

        # Splitting
        for ci in eachindex(cs)
            M = length(cs[ci].indices)
            C = length(cs[ci].consSets)
            (M == 0 || C == 0 || M == 1) && continue

            mask = map(ce -> ce.isConstant ? ce.value : 0, cs[ci].consSets[1])
            for set in cs[ci].consSets
                for ix in 1:M
                    if (!set[ix].isConstant || mask[ix] != set[ix].value) && mask[ix] != 0
                        mask = setNth(mask, UInt64(ix), UInt64(0))
                    end
                end
            end
            sum(mask) == 0 && continue
            mask = map(x -> x != 0 ? 0 : 1, mask)
            should_stop = false
            #println("Applying splitting for ", cs[ci])
            for ix in 1:M
                if mask[ix] == 0
                    push!(cs, Constraint{1,1}((cs[ci].indices[ix],), ((cs[ci].consSets[1][ix],),)))
                end
            end
            
            # map!(set -> set[filter(i -> values[i]==0, eachindex(set))], cs[ci].consSets, cs[ci].consSets)
            # cs[ci].indices = cs[ci].indices[filter(i -> values[i]==0, eachindex(cs[ci].indices))]
            
            cs[ci] = Constraint(
                filter(
                    x -> x != 0, 
                    cs[ci].indices .* mask
                ),
                map(
                    set -> filter(
                        ce -> !(ce == ConstraintElem(true, 0)),
                        ntuple(i -> mask[i] == 0 ? ConstraintElem(true, 0) : set[i], M)
                    ), 
                    cs[ci].consSets
                )
            )

            #print_constraints(cs)
        end

        # Simplification
        for ci in eachindex(cs)
            M = length(cs[ci].indices)
            C = length(cs[ci].consSets)

            (M == 0 || C == 1 || C == 0) && continue
            
            mask = ntuple(si -> begin
                for pi in eachindex(cs[ci].consSets)
                    pi == si && continue
                    if issubset(cs[ci].consSets[si], cs[ci].consSets[pi])
                        return 1 # delete
                    end
                end
                return 0 # keep
            end, C)
            sum(mask) == 0 && continue
            should_stop = false
            #println("Applying simplification on ", cs[ci])

            cs[ci] = Constraint(
                cs[ci].indices,
                filter(
                    set -> !(set[1].isConstant && set[1].value == 0), 
                    ntuple(i -> mask[i] == 0 ? cs[ci].consSets[i] : ntuple(j -> ConstraintElem(true, 0), M), C)
                )
            )
        
            #print_constraints(cs)
        end

        # Remove illegal or redundant constraints
        filter!(c -> length(c.indices) != 0, cs)
        if count(c -> length(c.consSets) == 0, cs) != 0
            indices = findall(i -> length(cs[i].consSets) == 0, eachindex(cs))
            for ix in indices
                println("Shapes of expressions ", map(i -> nodes[div(i-1, N)+1], cs[ix].indices), " do not match")
                println("Their determined shapes are: ", map(i -> solved[(div(i-1, N)*N+1):(div(i-1, N)*N+N)], cs[ix].indices))
            end
            @warn "Expression cannot rigurously exist"
            return false
        end


    end

    println("Finished")
    print_constraints(cs)

end