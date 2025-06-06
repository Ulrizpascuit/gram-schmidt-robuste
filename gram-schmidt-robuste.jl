using LinearAlgebra

function gram_schmidt_robuste(A::AbstractMatrix; atol=1e-12)
    n, m = size(A)
    Q = zeros(n, m)
    nb_vects = 0
    for j in 1:m
        v = A[:,j]
        for i in 1:nb_vects
            v -= (Q[:,i]' * A[:,j]) * Q[:,i]
        end
        norm_v = norm(v)
        if norm_v < atol
            @warn "Colonne $j ignorée (dépendance linéaire ou vecteur trop petit, norm = $norm_v)"
            continue
        end
        nb_vects += 1
        Q[:,nb_vects] = v / norm_v
    end
    return Q[:, 1:nb_vects]
end
