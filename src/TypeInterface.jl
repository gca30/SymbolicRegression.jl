module TypeInterfaceModule

using Random: AbstractRNG

get_base_type(::Type{W}) where {W <: Real} = W
get_base_type(::Type{Complex{BT}}) where {BT} = get_base_type(BT)

function mutate_value(rng::AbstractRNG, maxFactor:: BT, val::T) where {BT, T} end

function mutate_value(rng::AbstractRNG, maxFactor:: BT, val::T) where {BT,T<:Real}
    negate :: Bool = false
    if maxFactor < 0
        maxFactor = -maxFactor
        negate = true
    end
    retval = val
    factor = convert(T, maxFactor^rand(rng, T))
    makeConstBigger = rand(rng, Bool)
    if makeConstBigger
        retval *= factor
    else
        retval /= factor
    end
    if negate
        retval *= convert(T, -1)
    end
    return retval
end

@inline function mutate_value(rng::AbstractRNG, maxFactor:: BT, val::T) where {BT,Q,T<:Complex{Q}}
    Complex(mutate_value(rng, maxFactor, real(val)), mutate_value(rng, maxFactor, imag(val)))
end

end