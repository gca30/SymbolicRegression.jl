#TODO - eventually move some of these
# into the SR call itself, rather than
# passing huge options at once.

function binopmap(op)
    if op == plus
        return +
    elseif op == mult
        return *
    elseif op == sub
        return -
    elseif op == div
        return /
    elseif op == ^
        return pow
    end
    return op
end

function unaopmap(op)
    if op == log
        return log_abs
    elseif op == log10
        return log10_abs
    elseif op == log2
        return log2_abs
    elseif op == sqrt
        return sqrt_abs
    end
    return op
end

struct Options{A,B}

    binops::A
    unaops::B
    bin_constraints::Array{Tuple{Int,Int}, 1}
    una_constraints::Array{Int, 1}
    ns::Int
    parsimony::Float32
    alpha::Float32
    maxsize::Int
    maxdepth::Int
    fast_cycle::Bool
    migration::Bool
    hofMigration::Bool
    fractionReplacedHof::Float32
    shouldOptimizeConstants::Bool
    hofFile::String
    npopulations::Int
    nrestarts::Int
    perturbationFactor::Float32
    annealing::Bool
    batching::Bool
    batchSize::Int
    mutationWeights::Array{Float64, 1}
    warmupMaxsize::Int
    limitPowComplexity::Bool
    useFrequency::Bool
    npop::Int
    ncyclesperiteration::Int
    fractionReplaced::Float32
    topn::Int
    verbosity::Int
    probNegate::Float32
    nuna::Int
    nbin::Int
    seed::Union{Int, Nothing}

end

function Options(;
    binary_operators::NTuple{nbin, Any}=(div, plus, mult),
    unary_operators::NTuple{nuna, Any}=(exp, cos),
    bin_constraints=nothing,
    una_constraints=nothing,
    ns=10, #1 sampled from every ns per mutation
    topn=10, #samples to return per population
    parsimony=0.000100f0,
    alpha=0.100000f0,
    maxsize=20,
    maxdepth=nothing,
    fast_cycle=false,
    migration=true,
    hofMigration=true,
    fractionReplacedHof=0.1f0,
    shouldOptimizeConstants=true,
    hofFile=nothing,
    npopulations=nothing,
    nrestarts=3,
    perturbationFactor=1.000000f0,
    annealing=true,
    batching=false,
    batchSize=50,
    mutationWeights=[10.000000, 1.000000, 1.000000, 3.000000, 3.000000, 0.010000, 1.000000, 1.000000],
    warmupMaxsize=0,
    limitPowComplexity=false,
    useFrequency=false,
    npop=1000,
    ncyclesperiteration=300,
    fractionReplaced=0.1f0,
    verbosity=convert(Int, 1e9),
    probNegate=0.01f0,
    seed=nothing
   ) where {nuna,nbin}

    if hofFile == nothing
        hofFile = "hall_of_fame.csv" #TODO - put in date/time string here
    end

    if una_constraints == nothing
        una_constraints = [-1 for i=1:nuna]
    end
    if bin_constraints == nothing
        bin_constraints = [(-1, -1) for i=1:nbin]
    end

    if maxdepth == nothing
        maxdepth = maxsize
    end

    if npopulations == nothing
        npopulations = nworkers()
    end

    binary_operators = map(binopmap, binary_operators)
    unary_operators = map(unaopmap, unary_operators)

    mutationWeights = map((x,)->convert(Float64, x), mutationWeights)
    if length(mutationWeights) != 8
        error("Not the right number of mutation probabilities given")
    end

    for (op, f) in enumerate(map(Symbol, binary_operators))
        _f = if f == Symbol(pow)
            Symbol(^)
        else
            f
        end
        if !isdefined(Base, _f)
            continue
        end
        @eval begin
            Base.$_f(l::Node, r::Node)::Node = (l.constant && r.constant) ? Node($f(l.val, r.val)::AbstractFloat) : Node($op, l, r)
            Base.$_f(l::Node, r::AbstractFloat)::Node =        l.constant ? Node($f(l.val, r)::AbstractFloat)     : Node($op, l, r)
            Base.$_f(l::AbstractFloat, r::Node)::Node =        r.constant ? Node($f(l, r.val)::AbstractFloat)     : Node($op, l, r)
        end
    end

    for (op, f) in enumerate(map(Symbol, unary_operators))
        if !isdefined(Base, f)
            continue
        end
        @eval begin
            Base.$f(l::Node)::Node = l.constant ? Node($f(l.val)::AbstractFloat) : Node($op, l)
        end
    end

    Options{typeof(binary_operators),typeof(unary_operators)}(binary_operators, unary_operators, bin_constraints, una_constraints, ns, parsimony, alpha, maxsize, maxdepth, fast_cycle, migration, hofMigration, fractionReplacedHof, shouldOptimizeConstants, hofFile, npopulations, nrestarts, perturbationFactor, annealing, batching, batchSize, mutationWeights, warmupMaxsize, limitPowComplexity, useFrequency, npop, ncyclesperiteration, fractionReplaced, topn, verbosity, probNegate, nuna, nbin, seed)
end

