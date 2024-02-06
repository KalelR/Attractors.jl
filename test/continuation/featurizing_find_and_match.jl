using Revise
using Attractors, Test

function dumb_map(dz, z, p, n)
    x,y = z
    r = p[1]

    if r < 0.5
        dz[1] = dz[2] = 0.0
    else
        if x > 0
            dz[1] = r
            dz[2] = r
        else
            dz[1] = -r
            dz[2] = -r
        end
    end
    return
end

r = 1.0
ds = DeterministicIteratedMap(dumb_map, [0., 0.], [r])
featurizer(A, t) = A[1]
grouping_config = GroupViaPairwiseComparison(distance_threshold=0.5)
mapper = AttractorsViaFeaturizing(ds, featurizer, grouping_config) 

og_ics = Dataset([[1.0, 1.0], [1.0, 1.0], [1.0,1.0], [-1.0,-1.0]]); additional_ics = Dataset([[-1.0,-1.0]])  #OK!
og_ics = Dataset([[1.0, 1.0], [1.0, 1.0], [1.0,1.0]]); additional_ics = Dataset([[-1.0,-1.0], [-1.0,-1.0]]) 

fs, labels = basins_fractions(mapper, og_ics; additional_ics)

@test fs == Dict(2 => 0.4, 1 => 0.6)
@test labels == [1, 1, 1, 2, 2]

