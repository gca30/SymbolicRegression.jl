
# Second try at making a shape inference algorithm
# First iteration supported binary operator broadcasting, unary operator broadcasting, 
#   permutedims operation (transpose), cross products, etc..
# Second iteration should support concatenation, convolution, 
#  ^ These operation require that the dimensions are added or subtracted between eachother
#    The representation will be the same: list of constraints of the form 
#       (tuple of dimension) = union of sets of possible tuples formed by constants and shape variables
#       this time, there can also be arithmetic operations in the tuple or the parameters, leading to each constraint
#       element being a linear combination of variables and constants

#=
const LinearCombination{V, T} = NTuple{V, Tuple{Int8, T}}

Base.show(io, lc::LinearCombination{V,T}) where {V,T} = begin
    printed = false
    for i in 1:V
        if lc[i][1] == 1
            if printed print(io, " + ") end
            print(io, lc[i][2])
            printed = true
        elseif lc[i][1] == -1
            if printed print(io, " - ") else print(io, "-") end
            print(io, lc[i][2])
            printed = true 
        elseif lc[i][1] != 0
            if printed 
                if lc[i][1] < 0 print(io, " - $(-lc[i][1])") else print(io, " + $(lc[i][1])") end 
            else
                if lc[i][1] < 0 print(io, "-$(-lc[i][1])") else print(io, "$(lc[i][1])") end 
            end
            print(io, lc[i][2])
            printed = true
        end
    end
end

@inline Base.map(f, lc::LinearCombination{V,T}) where {V,T} = ntuple(i -> (lc[i][1], f(lc[i][2])), Val(V))
@inline Base.isempty(lc::LinearCombination{V,T}) where {V,T} = all(((coef, val)) -> coef == 0 || val == zero(val), lc)

=# 

#=
print_parensd_tuple(a::NTuple{N,C}) = begin
end

@inline Base.isempty(c::Constraint{M,C,Va,Vn}) where {M,C,Va,Vn} = M == 0 || all(isempty, c.lhs)

Base.show(io, c::Constraint{M,C,Va,Vn}) where {M,C,Va,Vn} = begin
    
end

# Vn = 3
# C = 3
# M = 6



# example of vertical concatenation
# (a1, a2) concat (a3, a4) -> (a5, a6)
#   (a2, a4, a6) elem {(n,n,n)}
#   (a5, a1, a3) elem {(n,m, n+m)}

# For example, if we get a constraint of the form:
#   a3 elem 5
#   (a5, a1) elem {(n+5,n)}   # all ns are replaced with n+5 now, except

=#

using StaticArrays

# M - tuple size
# A - number of variables
# C - number of sets
# V - number of variables in set + 1 (as the first row is constants)
# last value of set is 1
# we lose type sfaety because it's too much pain to deal with
mutable struct Constraint{M, C, V}
    tuple :: MVector{M, Int32}
    sets :: MVector{C, MMatrix{M, V, Int32}}
end

@inline Base.zero(::Type{Constraint}) = Constraint{0,0,0,0}(zero(MVector{0, Int32}), zero(MVector{0, MMatrix{0,0,Int32}}))
@inline isconstant(c::Constraint{M,A,C,V}) where {M,A,C,V} = (M,C,A,V) == (1,1,1,1) && c.tuple[1] != 0

cs = Constraint[]

# a1 = a2 is equivalent to:
push!(cs, Constraint{2,1,2}(
    MVector{2,Int32}([1, 2]),
    (MMatrix{2,2,Int32}([0 1; 0 1]),)
))

# a3 = 5 is equivalent to:
push!(cs, Constraint{1,1,1,1}(
    MMatrix{1,1,Int32}([1;;]),
    MVector{1,Int32}([3]),
    (MMatrix{1,1,Int32}([5;;]),)
))

# (a4,a5,a6) = {(n,1,n)} u {(n,n,n)} u {(n,n,1)} is:
push!(cs, Constraint{3,3,2}(
    MVector{3,Int32}([4, 5, 6]),
    (MMatrix{3,2}([0 1; 1 0; 0 1]), MMatrix{3,2}([0 1; 0 1; 0 1]), MMatrix{3,2}([0 1; 0 1; 1 0]))
))

# a9 = a7+a8 is:
push!(cs, Constraint{2,1,3}(
    MVector{3,Int32}([7, 8, 9]),
    (MMatrix{2, 3}([0 1 0; 0 1 0; 0 1 1]),)
))

# the outside is stored as:

using SparseArrays

# 1.  the expected number of variables
# 2.  the number of as

mutable struct CombinedConstraints
    # values :: SparseMatrixCSC{Int32, Int32}
    # var_occupied :: Vector{Int32}
    # ax_occupied :: BitVector
    values :: Matrix{Int32}

    CombinedConstraints(A) = new(zeros(Int32, min(A, 10), A))
end
@inline varvecsize(sb::CombinedConstraints) = size(cb.values, 1)

cb = CombinedConstraints(44) # SparseMatrixCSC{Int32, Int32}(undef, 10, 44)

expand(mat::Matrix) = cat(mat, zeros(size(mat)...); dims=1)

expand(cb::CombinedConstraints) = begin
    # cb2 = SparseMatrixCSC(Int32, Int32)(undef, size(cb.values, 1)*2, size(cb.values, 2))
    # @view(cb2[axes(cb, 1), axes(cb, 2)]) .= cb.values
    # cb.values = cb2
    # cb.values(findnz(cb.values)..., size(cb.values, 1)*2, size(cb.values, 2))
    # append!(cb.var_occupied, zeros(length(cb.var_occupied)))
    cb.values = expand(cb.values)
end

# expand(spvec::SparseVector{T1,T2}, size) where {T1,T2} = sparse(findnz(spvec)..., size)

first_unused_var!(cb::CombinedConstraints) = begin
    for i in axes(cb.values, 1)
        if i == 1 continue end
        if all(==(0), @view(cb.values[i,:])) return i end
    end
    v = length(cb.occupied) + 1
    expand(cb)
    return v
end

# @inline is_permutation_matrix(c::MMatrix{N,N,T}) where {N,T} = all(==(1), sum(c; dims=1)) && all(==(1), sum(c; dims=2)) 
# @inline is_spvec_empty(spvec) = length(spvec.nzind) == 0
@inline should_outsubst(::Constraint{M,C,V}) where {M,C,V} = C == 1

function solveDioph(cb::CombinedConstraints, dec::SparseVector{Int32, Int32})

end

#= ::Tuple{Int32,Int32,Int32} -> if an error occurs, it returns the upsetting "error code", a_x and value =#
function outsubst(c::Constraint{M,A,C,V}, cb::CombinedConstraints) where {M,A,C,V}
    currentVars = varvecsize(cb)
    # create the map for consts
    constsMap = zeros(Int32, V, currentVars)
    # constsMap = MVector{V, SparseVector{Int32, Int32}}(ntuple(Returns(SparseVector{Int32, Int32}(undef, currentVars)), Val(V)))
    for i in 1:M
        ax = c.tuple[i]
        value = c.sets[1][i,:]
        @assert length(value) == V
        if cb.ax_occupied[ax]
            # cbvalue =  # value which must equal the other constants
            
            # both constants case
            if V == 1 && cb.values[ax,1] != 0 && nnz(cb.values[ax,:]) == 1 && value[1] != cb.values[ax,1]
                return 1, ax, value[1], false
            end

            # adding the constants to the array
            for j in 2:V
                if value[j] == 0 continue end
                if is_spvec_empty(constsMap[j])
                    newvar = first_unused_var!(cb)
                    if newvar > currentVars
                        currentVars = length(cb.var_occupied)
                        map!(spvec -> expand(spvec, currentVars), constsMap, constsMap)
                    end
                    constsMap[j][newvar] = 1
                    cb.var_occupied[newvar] += 1
                end
            end

            # computing the Diophantine equation constraint
            result = SparseVector{Int32, Int32}(undef, currentVars)
            if value[1] != 0
                result[1] += value[1]
            end
            for j in 2:V
                if value[j] == 0 continue end
                @. result += constsMap[j] * value[j]
            end
            @. result += cb.values[ax,:]
            

            solveDioph(cb, result)
            # the constraint is now : result = 0


            # the set of equations are:
            #   n_x = lc(p_x) + c_x/
            #   lc(n_x)+c1 = lc(p_x) + c2 => lc(p_x)+c3 = lc(p_x) + c2 => lc(p_x) = c4
            #   

            # cbnvars = length(cbvalue.nzind)
            # if cbvalue[1] != 0
            #     cbnvars -= 1
            # end
            # vnvars = count(j -> value[j] != 0 && j != 1, eachindex(value))
            # # things get complicated when we are working with integers
            # # https://en.wikipedia.org/wiki/Diophantine_equation#System_of_linear_Diophantine_equations
            # if vnvars > cbvars
            #     # more cb variables than vnvariables
            #     # there will be variables left in cb
            # elseif vnvars == cbnvars
            #     # there is a one-to-one mapping between new variables and old variables
            # else
            #     # more vn variables than vnvariables
            #     # we will need to add new vars to cb
            # end
            
        else
            # result = SparseVector{Int32, Int32}(undef, currentVars)
            cb.ax_occupied[ax] = true
            if value[1] != 0
                cb.values[ax,1] += value[j]
                continue
            end
            for j in 2:V
                if value[j] == 0 continue end                
                if is_spvec_empty(constsMap[j])
                    # get a new variable, extend if necesary
                    newvar = first_unused_var!(cb)
                    if newvar > currentVars
                        currentVars = length(cb.var_occupied)
                        map!(spvec -> expand(spvec, currentVars), constsMap, constsMap)
                    end
                    constsMap[j][newvar] = 1
                end

                # warning: row iteration
                for k in 1:length(constsMap[j])
                    if constsMap[j][k] == 0 continue end
                    if cb.values[ax,k] == 0
                        cb.var_occupied[k] += 1
                    end
                    cb.values[ax,k] += constsMap[j][k]
                end
                    
            end
            
        end
    end
    return 0, 0, 0
end

function shape_inference_iteration(cs::Vector{Constraint}, cb::CombinedConstraints)
    for ci in eachindex(cs)
        if should_outsubst(cs[ci])
            outsubst(cs[ci], cb)
            cs[ci] = zero(Constraint)
        end
    end
    # filter(!isempty, cs)
end