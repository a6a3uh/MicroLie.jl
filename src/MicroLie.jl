module MicroLie

using LinearAlgebra
using LabelledArrays

export QUnit, QUnitT, qfields, qmatfields
export qleft, qright
export spin, unspin
export adj, qExp, qLog, RLog, jright
export (⊕), (⊗), (⊖)
export euler213

    
qfields = (re=1, i=2, j=3, k=4, im=2:4, q=1:4)
qmatfields = (re=(1:1,:), i=(2:2,:), j=(3:3,:), k=(4:4,:), im=(2:4,:), q=(1:4,:))
const QUnit = @SLVector qfields
const QUnitT{T} = @SLVector T qfields

qleft(q::AbstractVector) = qleft(QUnit(q))
qright(q::AbstractVector) = qright(QUnit(q))
qLog(q::AbstractVector) = qLog(QUnit(q))
adj(q::AbstractVector) = adj(QUnit(q))
⊗(p::AbstractVector, q::AbstractVector) = QUnit(p) * QUnit(q)
⊕(p::AbstractVector, w::AbstractVector) = QUnit(p) + w
⊖(p::AbstractVector, q::AbstractVector) = QUnit(p) - QUnit(q)

LinearAlgebra.normalize(q::QUnitT) = QUnit(sign(q.re) * normalize(Vector(q)))

qleft(q::QUnitT) = I(4) * q.re + [0 -q.im'; q.im spin(q.im)]
qright(q::QUnitT) = I(4) * q.re + [0 -q.im'; q.im -spin(q.im)]

spin(v::AbstractVector{<:Real}) =
    [0 -v[3] v[2]
     v[3] 0 -v[1]
     -v[2] v[1] 0]

unspin(m::AbstractMatrix) = [m[3,2], m[1,3], m[2,1]]

function qExp(w::AbstractVector{T}) where T
    Θ = norm(w)
    if Θ != 0
        u = w / Θ
        re = cos(Θ / 2)
        im = sin(Θ / 2) * u
    else
        re = 1
        im = zeros(3)
    end
    return QUnit([re; im])
end

function qLog(q::QUnitT)
    n = norm(q.im)
    if n == 0
        return zeros(3)
    end
    u = q.im / n
    θ = atan(n, q.re)
    
    v = 2 * θ * u
    return norm(v) < π ? v : (norm(v) - 2π) * normalize(v)
end

function adj(q::QUnitT)
    q.im * q.im' + (I(3)q.re + spin(q.im))^2
end


function Base.:*(p::QUnitT{T}, q::QUnitT{T})::QUnitT{T} where T
    QUnitT{T}([p.re * q.re - p.im' * q.im; p.im * q.re + p.re * q.im + p.im × q.im]) 
end

function Base.:+(q::QUnitT{T}, θ::AbstractVector{T})::QUnitT{T} where T
    q * qExp(θ)
end

function Base.:-(p::QUnitT{T}, q::QUnitT{T})::AbstractVector{T} where T
    # p = q ⊕ θ
    # θ = p ⊖ q
    qLog(conj(q) * p)
end

function Base.conj(q::QUnitT{T}) where T
    QUnitT{T}([q.re; -q.im])
end

function RLog(w::AbstractVector)
    a = norm(w)
    if a != 0
        u = w / a
        R = cos(a)I(3) + sin(a)spin(u) + (1-cos(a))u*u'
    else
        R = Matrix{eltype(w)}(I(3))
    end
    return R
end

function jright(x::AbstractVector)::AbstractMatrix
    θ = norm(x)
    if θ ≈ 0
        return I(3)
    else
        return I(3) - (1-cos(θ)) / θ ^ 2 * spin(x) + (1-sin(θ)) / θ ^ 3  * spin(x) ^ 2
    end
end

euler213(r::Matrix) = [atan(-r[1,3], r[3,3]), asin(r[2,3]), atan(-r[2,1], r[2,2])] 

end # module MicroLie
