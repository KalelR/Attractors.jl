include("grouping/all_grouping_configs.jl")

struct AttractorsViaComparing{DS<:DynamicalSystem, G<:GroupingConfig, F, T, D, V} <: AttractorMapper
    ds::DS
    featurizer::F
    comparing_config::G
    Ttr::T
    Δt::T
    total::T
    threaded::Bool
    attractors::Dict{Int, StateSpaceSet{D, V}}
end

function AttractorsViaComparing(ds::DynamicalSystem, featurizer::Function,
    comparing_config::ComparingConfig = GroupViaClustering();
    T=100, Ttr=100, Δt=1, threaded = true,
)
D = dimension(ds)
V = eltype(current_state(ds))
# For parallelization, the dynamical system is deepcopied.
return AttractorsViaComparing(
    ds, featurizer, comparing_config, Ttr, Δt, T, threaded, Dict{Int, StateSpaceSet{D,V}}(),
)
end

function basins_fractions(mapper::AttractorsViaComparing, og_ics::ValidICS;
    show_progress = true, N = 1000, additional_ics::Union{ValidICS, Nothing} = nothing,
)
    ics = 
        if isnothing(additional_ics) || typeof(og_ics) <: Function
            og_ics
        else
            Dataset(vcat(Matrix(og_ics), Matrix(additional_ics)))
        end

    features = featurize_and_compare(mapper, ics; show_progress, N)
    @show features
    # @info "Finished features"
    GC.gc()
    # @info "Finished features, now going to clustering"
    ufeats = unique(map(feature->round.(feature, sigdigits=2), features))
    # @show ufeats
    group_labels = group_features(features, mapper.group_config)
    # @info "Finished clustering"
    fs = basins_fractions(group_labels) # Vanilla fractions method with Array input

    if typeof(ics) <: AbstractStateSpaceSet
    # @info "Extracting atts"
    fs = basins_fractions(group_labels) # Vanilla fractions method with Array input
        attractors = extract_attractors(mapper, group_labels, ics)
        overwrite_dict!(mapper.attractors, attractors)
        return fs, group_labels
    else #no attractor extraction if `ics` are a sampler function
        return fs
    end
end