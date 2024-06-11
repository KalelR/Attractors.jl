"""
Re-integrate attractors, giving new T, Ttr and Δt, regroup them, and create a new `fs`
that accounts for any changes in the attractors from this new grouping. 

Useful in eliminating duplicate attractors, which may come about from unideal simulation
 times (e.g. to elimiate long transients). 
"""
function integrate_and_regroup(mapper, atts, fs; T=100, Ttr=100, Δt=1, threaded = true)
    ics = Dict(k => att[end] for (k, att) in atts)
    atts_integ, ts = integrate_ics(mapper.ds, ics, T; threaded, Ttr, Δt)
    features = Dict(k => mapper.featurizer(att, ts) for (k, att) in atts_integ) 
    @show features
    labels = group_features(collect(values(features)), mapper.group_config)
    tmap = Dict(keys(features) .=> labels) # map keys from previous grouping to keys from new grouping
    new_keys = unique(labels) 
    fs_new, atts_new = _updated_fs_and_atts(fs, atts_integ, new_keys, tmap)
    return atts_new, fs_new, features
end

"""
maintains the key order from `ics`
"""
function integrate_ics(ds, args...; threaded=true, kwargs...)
    if !(threaded)
        integrate_ics_single(ds, args...; kwargs...)
    else
        integrate_ics_threaded(ds, args...; kwargs...)
    end
end

function integrate_ics_single(ds, ics::Dict{A, SVector{B, C}}, T; Ttr=0.0, Δt=1.0, show_progress=true) where {A, B, C}
    N = length(ics) # number of actual ICs
    att_dict = Dict{A, StateSpaceSet{B,C}}()
    progress = ProgressMeter.Progress(N; desc = "Integrating trajectories:", enabled=show_progress)
    for k in collect(keys(ics))
        ic = ics[k]
        att_dict[k] = trajectory(ds, T, ic; Ttr, Δt)[1]
        ProgressMeter.next!(progress)
    end
    ts = Ttr:Δt:Ttr+T
    return att_dict, ts 
end

function integrate_ics_threaded(ds, ics::Dict{A, SVector{B, C}}, T; Ttr=0.0, Δt=1.0, show_progress=true) where {A, B, C}
    N = length(ics) # number of actual ICs
    systems = [deepcopy(ds) for _ in 1:(Threads.nthreads() - 1)]
    pushfirst!(systems, ds)
    
    att_dict = Dict{A, StateSpaceSet{B,C}}()
    
    progress = ProgressMeter.Progress(N; desc = "Integrating trajectories:", enabled=show_progress)
    Threads.@threads for k in collect(keys(ics))
        ic = ics[k]
        ds = systems[Threads.threadid()]
        att_dict[k] = trajectory(ds, T, ic; Ttr, Δt)[1]
        ProgressMeter.next!(progress)
    end
    ts = Ttr:Δt:Ttr+T
    return att_dict, ts 
end

# function transition_info(group_labels, features)
#     idxs_going_to_each_key = map(ulab->findall(x->x==ulab, group_labels), new_keys)
#     return new_keys, tmap, idxs_going_to_each_key
# end

function _updated_fs_and_atts(fs::Dict{A,B}, atts_integ::Dict{A, C}, new_keys, tmap) where {A,B,C}
    fs_new = Dict{A,B}()
    atts_new = Dict{A,C}()
    for new_key in new_keys 
        prev_keys_going_to_new_key = [k for (k,v) in tmap if v == new_key]
        fs_prev_keys = map(k->fs[k], prev_keys_going_to_new_key)
        fs_new[new_key] = sum(fs_prev_keys)
        
        prev_keys_first = minimum(prev_keys_going_to_new_key)
        # @show prev_keys_first
        atts_new[new_key] = atts_integ[prev_keys_first]
    end
    return fs_new, atts_new
end

function integrate_and_regroup_and_rematch(fam, prange, pidx, atts_info, fs_curves; kwargs...)
    fs_curves_new = deepcopy(fs_curves) 
    atts_info_new = deepcopy(atts_info)
       
    (; mapper, distance, threshold) = fam
    
    reset!(mapper)
    
    for idx in eachindex(prange)
        p = prange[idx]
        @info "In integrate and regroup, p = $p."
        set_parameter!(mapper.ds, pidx, p)
        atts, fs = atts_info_new[idx], fs_curves_new[idx]
        atts_new, fs_new = integrate_and_regroup(mapper, atts, fs; kwargs...)
        atts_info_new[idx] = atts_new 
        fs_curves_new[idx] = fs_new 
    end

    match_continuation!(fs_curves_new, atts_info_new; distance, threshold)
    return atts_info_new, fs_curves_new
end


function integrate_and_regroup_and_rematch_flow(fam, prange, pidx, atts_info, fs_curves; kwargs...)
    fs_curves_new = deepcopy(fs_curves) 
    atts_info_new = deepcopy(atts_info)
       
    (; mapper, distance, threshold) = fam
    
    reset!(mapper)
    
    for idx in eachindex(prange)
        p = prange[idx]
        @info "In integrate and regroup, p = $p."
        set_parameter!(mapper.ds, pidx, p)
        atts, fs = atts_info_new[idx], fs_curves_new[idx]
        atts_new, fs_new = integrate_and_regroup(mapper, atts, fs; kwargs...)
        atts_info_new[idx] = atts_new 
        fs_curves_new[idx] = fs_new 
    end

    match_continuation!(fs_curves_new, atts_info_new; distance, threshold)
    return atts_info_new, fs_curves_new
end

# fs = Dict(1=> 0.1, 3=>0.2, 5=>0.2, 2=>0.5)
# atts = Dict(1=>[1,1], 3=>[2,2], 5=>[3,3], 2=>[4,4])
# 
# atts_integ = Dict(1=>[1,1], 3=>[2,2], 5=>[3,3], 2=>[1,1])
# labels = replace(collect(keys(atts_integ)), 2=>1)
# 
# tmap = Dict(keys(atts_integ) .=> labels) # map keys from previous grouping to keys from new grouping
# new_keys = unique(labels) 
# @show tmap
# # @show idxs_going_to_each_key
# fs_new, atts_new = _updated_fs_and_atts(fs, atts_integ, new_keys, tmap)


# function _updated_atts(tmap, atts_integ)
#     # idxs_new_atts = [idx[1] for idx in idxs_going_to_each_key]
#     # @show idxs_new_atts
#     atts_new = Dict(new_key => atts_integ[old_key] for (old_key, new_key) in tmap) #each label is associated to an attractor; that attractor, amongs potentially many that lead to the same labels, is the one that appears first in collect(values(atts_integ)) (a bit arbitrary but ok)
#     return atts_new
# end
# 
# function _updated_atts(new_keys, idxs_going_to_each_key, atts_integ)
#     idxs_new_atts = [idx[1] for idx in idxs_going_to_each_key]
#     @show idxs_new_atts
#     atts_new = Dict(new_keys .=> collect(values(atts_integ))[idxs_new_atts]) #each label is associated to an attractor; that attractor, amongs potentially many that lead to the same labels, is the one that appears first in collect(values(atts_integ)) (a bit arbitrary but ok)
#     return atts_new
# end
# 
# function _updated_atts(new_keys, idxs_going_to_each_key, atts_integ)
#     idxs_new_atts = [idx[1] for idx in idxs_going_to_each_key]
#     atts_integ = collect(values(atts_integ)) #TODO: unecessary allocaton??
#     atts_new = Dict(new_keys .=> atts_integ[idxs_new_atts]) #each label is associated to an attractor; that attractor, amongs potentially many that lead to the same labels, is the one that appears first in collect(values(atts_integ)) (a bit arbitrary but ok)
#     return atts_new
# end

# function _updated_fs(fs::Dict{A,B}, new_keys, tmap) where {A,B}
#     fs_new = Dict{A,B}()
#     for key in new_keys 
#         prev_keys_going_to_new_key = [k for (k,v) in tmap if v == key]
#         fs_prev_keys = map(k->fs[k], prev_keys_going_to_new_key)
#         fs_new[key] = sum(fs_prev_keys)
#     end
#     return fs_new 
# end


# function integrate_and_regroup(mapper, atts, fs; T=100, Ttr=100, Δt=1, threaded = true, show_progress = false)
#     ics = StateSpaceSet([att[end] for (k, att) in atts])
#     features = Attractors.extract_features(mapper, ics; show_progress)
#     # features_dict = Dict(keys(atts) .=> features) #
#     labels = group_features(features, mapper.group_config)
#     
#     tmap = Dict(keys(atts) .=> labels) # map keys from previous grouping to keys from new grouping; ics and therefore features and therefore labels are in the order of keys(atts)
#     @show tmap
#     new_keys = unique(labels) 
#     
#     fs_new, atts_new = _updated_fs_and_atts(fs, atts_integ, new_keys, tmap)
#     return atts_new, fs_new
# end
