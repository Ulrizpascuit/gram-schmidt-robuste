# test.jl
using Test, LinearAlgebra
include("gram-schmidt.jl")
include("gram-schmidt-robuste.jl")

# Cas simple sans dépendance
A = [1 1; 0 1]
Q = gram_schmidt(A)
@test abs(norm(Q[:,1]) - 1) < 1e-12 # Norme colonne 1 =1
@test abs(norm(Q[:,2]) - 1) < 1e-12 # Norme colonne 2 = 1
@test abs(Q[:,1]' * Q[:,2]) < 1e-12 # Colonnes orthogonales

# Cas avec dépendance linéaire
A_dep = [1 2; 2 4]
@test_throws ErrorException gram_schmidt(A_dep) # Erreur pour gramschmidt non robuste

Q2 = gram_schmidt_robuste(A_dep)
@test size(Q2,2) == 1 # Une seule colonne survivante

# Cas tolérance
A_approx = [1 2+1e-13; 2 4+2e-13]
Q3 = gram_schmidt_robuste(A_approx, atol=1e-10)
@test size(Q3,2) == 1

println("Tous les tests sont passés.")
