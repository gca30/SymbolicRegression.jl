module SVMModule

# SVM (scalar, vector, matrix) - struct that contains all three datatypes
struct SVM{T}
    dims :: Int8 # number of dimmentions
    scalar :: T
    vector :: Vector{T}
    matrix :: Matrix{T}
    SVM{T}() where {T} = new(Int8(0), zero(T), T[], T[;;])
    SVM{T}(scalar :: W) where {T, W <: Number} = new(Int8(0), T(scalar), T[], T[;;])
    SVM{T}(vector :: Vector{W}) where {T, W <: Number} = new(Int8(1), zero(T), Vector{T}(vector) , T[;;])
    SVM{T}(matrix :: Matrix{W}) where {T, W <: Number} = new(Int8(2), zero(T), T[], Matrix{T}(matrix))
end

Base.one(::Type{W}) where {Q,W<:SVM{Q}} = return W(one(Q))

# get the base numerical type of a compound type
get_base_type(::Type{Complex{BT}}) where {BT} = BT
get_base_type(::Type{SVM{BT}}) where {BT} = return BT
get_base_type(::Type{W}) where {W <: Real} = return W


function Base.show(io::IO, x::SVM{T}) where {T}
    print(io, "SVM ")
    if x.dims == 0
        show(io, x.scalar)
    elseif x.dims == 1
        show(io, x.vector)
    elseif x.dims == 2
        show(io, x.matrix)
    end
end

function get_repr(s :: SVM{T}) where {T}
    if s.dims == 0
        s.scalar
    elseif s.dims == 1
        s.vector
    elseif s.dims == 2
        s.matrix
    end
end

macro svm_repr(r)
    return quote if r.dims == 0
        r.scalar
    elseif r.dims == 1
        r.vector
    elseif r.dims == 2
        r.matrix
    end end
end

# defining operators

for op in ((:(Base.:+), :.+), (:(Base.:-), :.-), (:(Base.:*), :.*), (:(Base.:/), :./), (:matmult, :*))
    @eval function $(op[1])(l::SVM{T}, r::SVM{T}) where {T}
        if l.dims == 0 && r.dims == 0
            SVM{T}(($(op[2]))(l.scalar, r.scalar))
        elseif l.dims == 1 && r.dims == 0
            SVM{T}(($(op[2]))(l.vector, r.scalar))
        elseif l.dims == 2 && r.dims == 0
            SVM{T}(($(op[2]))(l.matrix, r.scalar))
        elseif l.dims == 0 && r.dims == 1
            SVM{T}(($(op[2]))(l.scalar, r.vector))
        elseif l.dims == 1 && r.dims == 1
            SVM{T}(($(op[2]))(l.vector, r.vector))
        elseif l.dims == 2 && r.dims == 1
            SVM{T}(($(op[2]))(l.matrix, r.vector))
        elseif l.dims == 0 && r.dims == 2
            SVM{T}(($(op[2]))(l.scalar, r.matrix))
        elseif l.dims == 1 && r.dims == 2
            SVM{T}(($(op[2]))(l.vector, r.matrix))
        elseif l.dims == 2 && r.dims == 2
            SVM{T}(($(op[2]))(l.matrix, r.matrix))
        end
    end
end

function Base.transpose(s :: SVM{T}) where {T}
    if s.dims == 0
        SVM{T}(s.scalar)
    elseif s.dims == 1
        SVM{T}(s.vector)
    elseif s.dims == 2
        SVM{T}(transpose(s.matrix))
    end
end

function dot(l::SVM{T}, r::SVM{T}) where {T}
    if l.dims == 0 && r.dims == 0
        SVM{T}(l.scalar * r.scalar)
    elseif l.dims == 1 && r.dims == 1
        SVM{T}(sum(l.vector .* r.vector))
    elseif l.dims == 2 && r.dims == 2 && shape(l.matrix)[1] == shape(r.matrix)[1]
        SVM{T}(transpose(r.matrix) * l.matrix)
    else
        l # error
    end
end

function cross(l::SVM{T}, r::SVM{T}) where {T}
    if l.dims == 1 && r.dims == 1
        if size(l.vector)[1] == 2 && size(r.vector)[1] == 2
            return SVM{T}(l.vector[1]*r.vector[2] - l.vector[2]*r.vector[1])
        elseif size(l.vector)[1] == 3 && size(r.vector)[1] == 3
            return SVM{T}(l.vector .* r.vector) # substitute
        end
    end
    return l # error
end

end