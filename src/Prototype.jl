using MLJ
using SymbolicRegression
using SymbolicRegression.SVMModule

# Testing DynamicExpressions: works with SVM
# operators = GenericOperatorEnum(; binary_operators=[+, -, *, /, dot, cross, matmult], unary_operators=[])
# @extend_operators operators
#
# c1 = Node(SVM{Float64}; val = SVM{Float64}([1.0, 2.0, 0.5]))
# c2 = Node(SVM{Float64}; val = SVM{Float64}(1.0))
# c3 = Node(SVM{Float64}; val = SVM{Float64}(2.0))
# c4 = Node(SVM{Float64}; val = SVM{Float64}(3.0))
# c5 = Node(SVM{Float64}; val = SVM{Float64}([2.0 0 0; 0 2 0; 0 1 2]))
# x1 = Node(SVM{Float64}; feature=1)
# x2 = Node(SVM{Float64}; feature=2)
# x3 = Node(SVM{Float64}; feature=3)
# x4 = Node(SVM{Float64}; feature=4)
#
# #      |<------------------373--------------->| + [10, 11, 12]dot[20, 22, 35] = 1235
# tree = dot(x1 + cross(c1, x2), x3*c2) * c3 + c4 + dot(matmult(c5, x4), x4)
#
# X :: Vector{SVM{Float64}} = [ SVM{Float64}([1.0, 2, 3]), SVM{Float64}([4.0, 5, 6]), SVM{Float64}([7.0, 8, 9]), SVM{Float64}([10.0, 11, 12]) ]
# y :: SVM{Float64} = tree(X, operators)

# Running
function run_sr()
    
    model = SRRegressor(
        binary_operators=[+, -, *, /, dot, cross, matmult],
        unary_operators=[transpose],
        niterations=30
    )
    
    X :: Matrix{SVM{Float64}} = [SVM{Float64}(randn(3)) for _ in 1:100, _ in 1:5]
    y = [dot(cross(X[i, 1], SVM{Float64}([1.0, 2.0, 3.0])), X[i, 2]) + SVM{Float64}(4.0) for i in 1:100]

    mach = machine(model, X, y)
    fit!(mach)

    r = report(mach)
end