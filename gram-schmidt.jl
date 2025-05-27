# gram-schmidt.jl
"""
    gram_schmidt(A::AbstractMatrix)

Implémente la méthode classique de Gram-Schmidt pour orthonormaliser les colonnes de A.
Retourne une matrice Q dont les colonnes sont orthonormales.
"""
function gram_schmidt(A::AbstractMatrix)
    n, m = size(A)
    Q = zeros(eltype(A), n, m)
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
