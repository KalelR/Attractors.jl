export FeaturizingFindAndMatch, FFAM
import ProgressMeter
using Random: MersenneTwister

"""
    FeaturizingFindAndMatch <: AttractorsBasinsContinuation
    FeaturizingFindAndMatch(mapper::AttractorsViaFeaturizing; kwargs...)

A method for [`continuation`](@ref) as in [Datseris2023](@cite) that uses the featurizing algorithm for finding attractors ([`AttractorsViaFeaturizing`](@ref))
and the "matching attractors" functionality offered by [`match_continuation!`](@ref). Based heavily on the `RecurrencesFindAndMatch`, which uses the recurrences algorithm ([`AttractorsViaRecurrences`](@ref)).

You can use `FFAM` as an alias.

## Keyword arguments

- `distance = Centroid(), threshold = Inf, use_vanished = !isinf(threshold)`:
  propagated to [`match_continuation!`](@ref).
- `info_extraction = identity`: A function that takes as an input an attractor (`StateSpaceSet`)
  and outputs whatever information should be stored. It is used to return the
  `attractors_info` in [`continuation`](@ref). Note that the same attractors that
  are stored in `attractors_info` are also used to perform the matching in
  [`match_continuation!`](@ref), hence this keyword should be altered with care.
- `seeds_from_attractor`: A function that takes as an input an attractor and the number of
  initial conditions to sample from the attractor, and returns a vector with the sampled initial
  conditions. By default, we just select the initial conditions randomly from the attractor.

## Description

At the first parameter slice of the continuation process, attractors and their fractions
are found as described in the [`AttractorsViaFeaturizing`](@ref) mapper identifying groups
of features. At each subsequent parameter slice, initial conditions are seeded from the
 previously found attractors. They are put together with the pre-generated `ics` given to
 `continuation`. Then, all of them are passed onto [`basins_fractions`](@ref) for
 [`AttractorsViaFeaturizing`](@ref), which featurizes and groups them to identify the
 attractors. 

Naturally, during this step new attractors may be found. Once the basins fractions are
computed, the parameter is incremented again and we perform the steps as before.

This process continues for all parameter values. After all parameters are exhausted,
the found attractors (and their fractions) are "matched" to the previous ones.
I.e., their _IDs are changed_, so that attractors that are "similar" to those at a
previous parameter get assigned the same ID.
Matching is done by the [`match_continuation!`](@ref) function and is an _orthogonal_
step. This means, that if you don't like the initial
outcome of the matching process, you may call [`match_continuation!`](@ref) again
on the outcome with different matching-related keywords.
You do not need to compute attractors and basins again!

Matching is a very sophisticated process that can be understood in detail by reading
the docstrings of [`match_statespacesets!`](@ref) first and then [`match_continuation!`](@ref).
Here is a short summary: attractors from previous and current parameter are matched
based on their "distance". By default this is distance in state space, but any measure of
"distance" may be used, such as the distance between Lyapunov spectra.
Matching prioritizes new->old pairs with smallest distance: once these are matched
the next available new->old pair with smallest distance is matched, until all new/old
attractors have been matched. The `threshold` keyword establishes that attractors with
distance > `threshold` do not get matched.
Additionally, use `use_vanished = true` if you want to include as matching candidates
attractors that have vanished during the continuation process.
"""
struct FeaturizingFindAndMatch{A, M, R<:Real, S, E, I} <: AttractorsBasinsContinuation
    mapper::A
    distance::M
    threshold::R
    seeds_from_attractor::S
    info_extraction::E
    number_seeded_ics::I
end

"Alias for [`FeaturizingFindAndMatch`](@ref)."
const FFAM = FeaturizingFindAndMatch

function FeaturizingFindAndMatch(
        mapper::AttractorsViaFeaturizing; distance = Centroid(),
        threshold = Inf, seeds_from_attractor = _default_seeding_process_featurizing,
        info_extraction = identity, number_seeded_ics=nothing
    )

    if isnothing(number_seeded_ics)
        number_seeded_ics = typeof(mapper.group_config) <: GroupViaClustering ?
                            5*mapper.group_config.min_neighbors : 5
    end
    return FeaturizingFindAndMatch(
        mapper, distance, threshold, seeds_from_attractor, info_extraction, number_seeded_ics
    )
end

function _default_seeding_process_featurizing(attractor::AbstractStateSpaceSet, number_seeded_ics=10; rng = MersenneTwister(1))
    return [rand(rng, vec(attractor)) for _ in 1:number_seeded_ics] #might lead to repeated ics, which is intended for the continuation
end

function _endpoint_seeding_process_featurizing(attractor::AbstractStateSpaceSet, number_seeded_ics=10; rng = MersenneTwister(1))
    ics = [rand(rng, vec(attractor)) for _ in 2:number_seeded_ics] #might lead to repeated ics, which is intended for the continuation
    pushfirst!(ics, attractor[end])
    return ics
end


function continuation(
        fam::FeaturizingFindAndMatch,
        prange, pidx, ics;
        samples_per_parameter = 100, show_progress = true, keep_track_maximum=true,
    )
    progress = ProgressMeter.Progress(length(prange);
        desc="Continuating basins fractions:", enabled=show_progress
    )

    if ics isa Function
        error("`ics` needs to be a Dataset.")
    end

    (; mapper, distance, threshold, number_seeded_ics) = fam
    reset!(mapper)
    # first parameter is run in isolation, as it has no prior to seed from
    set_parameter!(mapper.ds, pidx, prange[1])
    fs, _ = basins_fractions(mapper, ics; show_progress = false)
    # At each parmaeter `p`, a dictionary mapping attractor ID to fraction is created.
    fractions_curves = [fs]
    # Furthermore some info about the attractors is stored and returned
    prev_attractors = deepcopy(extract_attractors(mapper))
    get_info = attractors -> Dict(
        k => fam.info_extraction(att) for (k, att) in attractors
    )
    info = get_info(prev_attractors)
    attractors_info = [info]
    ProgressMeter.next!(progress; showvalues = [("previous parameter", prange[1]),])
    alltime_maximum_key = maximum(keys(fs))
    # Continue loop over all remaining parameters
    for p in prange[2:end]
        @info "p = $p"
        set_parameter!(mapper.ds, pidx, p)
        reset!(mapper)
        
        # Collect ics from previous attractors to pass as additional ics to basins fractions (seeding).
        # To ensure that the clustering will identify them as clusters, we need to guarantee that there
        # are at least `min_neighbors` entries.
        additional_ics = Dataset(vcat(map(att-> 
            fam.seeds_from_attractor(att, number_seeded_ics),
            values(prev_attractors))...)) #dataset with ics seeded from previous attractors
        
        # for ic in additional_ics @show ic end
        idxs_special_additional_ics = 1:number_seeded_ics:length(additional_ics)
        # Now perform basin fractions estimation as normal, utilizing found attractors
        fs, _ = basins_fractions(mapper, ics;
            show_progress = false, additional_ics, idxs_special_additional_ics, prev_attractors
        )
        
        current_attractors = extract_attractors(mapper)
        push!(fractions_curves, fs)
        push!(attractors_info, get_info(current_attractors))
        overwrite_dict!(prev_attractors, current_attractors)
        ProgressMeter.next!(progress; showvalues = [("previous parameter", p),])
    end
    # Match attractors (and basins)
    match_continuation!(fractions_curves, attractors_info; distance, threshold)
    return fractions_curves, attractors_info
end

function reset!(mapper::AttractorsViaFeaturizing)
    empty!(mapper.attractors)
end

# Functions to perform matching by flow

function keys_minimum(d)
    vals = values(d)
    ks = keys(d)
    min = minimum(vals)
    idxs_min = findall(x->x==min, collect(vals))
    keys_min = collect(ks)[idxs_min] 
end

function _find_matching_att(featurizer, ts, att_future_int, atts_future) 
    f1 = featurizer(att_future_int, ts) #ts isn't correct here 
    dists_att_to_future = Dict(k=>evaluate(Euclidean(), f1, featurizer(att_future, ts)) for (k, att_future) in atts_future) 
    # @show dists_att_to_future
    label_nearest_atts = keys_minimum(dists_att_to_future)#TODO:thiswontdo-what-if-the-att-is-new??-what-if-the-same-key-is-matched-twice?
    # @show label_nearest_atts
    if isempty(label_nearest_atts)
        @error "WHAT"
    elseif length(label_nearest_atts) == 1 
        key_matching_att = label_nearest_atts[1] 
    else #coflowing attractors 
        @warn "WHAT"
    end
    return key_matching_att
end

function _find_coflowing(featurizer, ts, att_current, k_current, atts, coflowing_threshold) #coflowing if dist(att, atts) <= coflowing_th
    f1 = featurizer(att_current, ts)
    dist_to_atts = Dict(k=>evaluate(Euclidean(), f1, featurizer(att, ts)) for (k, att) in atts) #includes it to itself
    idxs_coflowing = findall(x->x<=coflowing_threshold, collect(values(dist_to_atts)))
    keys_coflowing = collect(keys(dist_to_atts))[idxs_coflowing]
    # @info "finding coflowing, $(dist_to_atts), keys = $(keys_coflowing), current att is $(att_current[end])"
    return keys_coflowing
end

function _key_legitimate(featurizer, ts, att_current, atts_prev)
    f1 = featurizer(att_current, ts)
    dist_to_prev = Dict(k=>evaluate(Euclidean(), f1, featurizer(att_prev, ts)) for (k, att_prev) in atts_prev)
    ks_legitimate = keys_minimum(dist_to_prev)
    # @info "finding legitimate, $(dist_to_prev), $ks_legitimate"
    if length(ks_legitimate) == 1 
        return ks_legitimate[1]
    else 
        # @warn "Two legitimates. Deal with it."
    end 
end

function replace_by_integrated_atts(ds, featurizer,atts_all, prange, pidx; T, Ttr, Δt, coflowing_threshold=0.1)
    all_keys = unique_keys(atts_all)
    atts_new = deepcopy(atts_all)
    for idx in 1:length(prange)-1
        p_future = prange[idx+1]
        ds_copy = deepcopy(mapper.ds)
        set_parameter!(ds_copy, pidx, p_future)
        atts_current = atts_new[idx]
        atts_future =  atts_new[idx+1]
        atts_future_integrated  = Dict(k=>trajectory(ds_copy, T, att_current[end]; Ttr, Δt)[1] for (k, att_current) in atts_current) 
        ts = Ttr:Δt:T
        keys_ilegitimate_all = Int64[]
        
        feats_current = Dict(k=>featurizer(att, ts) for (k, att) in atts_current)
        feats_future = Dict(k=>featurizer(att, ts) for (k, att) in atts_future)
        feats_future_int = Dict(k=>featurizer(att, ts) for (k, att) in atts_future_integrated)

        @info "idx = $idx, pcurrent = $(prange[idx]), pfuture = $p_future. feats_current = $feats_current, feats_future=$feats_future, feats_future_int=$feats_future_int"
        # @info "idx = $idx, pcurrent = $(prange[idx]), pfuture = $p_future." 
        for (k_future_int, att_future_int) in atts_future_integrated
            if k_future_int ∈ keys_ilegitimate_all 
                # @info "skipping $k_future_int because it is already found to be ilegitimate"
                continue
            end
            key_matching_att = _find_matching_att(featurizer, ts, att_future_int, atts_future)
            
            #before replacing, must check if att_future_int is legitimate (and doesn't come from an att that disappeared!)
            keys_coflowing = _find_coflowing(featurizer, ts, att_future_int, k_future_int, atts_future_integrated, coflowing_threshold)
            if length(keys_coflowing) == 1 #no coflowing, att is legitimate 
                att_replace = att_future_int 
                @info "Att key $k_future_int flows to single attractor $key_matching_att"
            else #coflowing 
                key_legitimate = _key_legitimate(featurizer, ts, att_future_int, atts_current)
                if key_legitimate == k_future_int #check if this is legitimate, i.e. if it is close to a previousy existing att 
                    att_replace = att_future_int 
                    keys_ilegitimate = filter(x->x==key_legitimate, keys_coflowing)
                    push!(keys_ilegitimate_all, keys_ilegitimate...)
                    @info "coflowing atts $keys_coflowing, with the legitimate being $key_legitimate."
                else  #"$k_future_int is coflowing but not legitimate"
                    continue
                end
            end
            
            atts_new[idx+1][key_matching_att] = att_replace 
        end
    end
    return atts_new 
end