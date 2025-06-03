using LinearAlgebra

function gram_schmidt(A::AbstractMatrix)
    n, m = size(A)
    Q = zeros(n, m)   
    for j in 1:m
        v = A[:,j]
        for i in 1:j-1
            v -= (Q[:,i]' * A[:,j]) * Q[:,i]
        end
        norm_v = norm(v)
        if norm_v == 0
            error("Dépendance linéaire détectée : le vecteur $j est nul après projection.")
        end
        Q[:,j] = v / norm_v
    end
    return Q
end
