"""
Re-integrate attractors, giving new T, Ttr and Δt, regroup them, and create a new `fs`
that accounts for any changes in the attractors from this new grouping. Useful in
eliminating duplicate attractors, which may come about from unideal simulation times (e.g.
to elimiate long transients).
#TODO: implementthreads
"""
function integrate_and_regroup(mapper, atts, fs; T=100, Ttr=100, Δt=1, threaded = true)
    atts_integ = Dict(k => trajectory(mapper.ds, T, att[end]; Ttr, Δt)[1] for (k, att) in atts)
    ts = Ttr:Δt:Ttr+T
    features = Dict(k => mapper.featurizer(att, ts) for (k, att) in atts_integ) 

    labels = group_features(collect(values(features)), mapper.group_config)
    new_keys, tmap, idxs_going_to_each_key = transition_info(labels, features)
    fs_new = _updated_fs(fs, new_keys, tmap)
    atts_new = _updated_atts(new_keys, idxs_going_to_each_key, atts_integ)
    return atts_new, fs_new,features
end

function transition_info(group_labels, features)
    tmap = Dict(collect(keys(features)) .=> group_labels) # map keys from previous grouping to keys from new grouping
    new_keys = unique(group_labels) 
    idxs_going_to_each_key = map(ulab->findall(x->x==ulab, group_labels), new_keys)
    return new_keys, tmap, idxs_going_to_each_key
end

function _updated_fs(fs::Dict{A,B}, new_keys, tmap) where {A,B}
    fs_new = Dict{A,B}()
    for key in new_keys 
        prev_keys_going_to_new_key = [k for (k,v) in tmap if v == key]
        fs_prev_keys = map(k->fs[k], prev_keys_going_to_new_key)
        fs_new[key] = sum(fs_prev_keys)
    end
    return fs_new 
end

function _updated_atts(new_keys, idxs_going_to_each_key, atts_integ)
    idxs_new_atts = [idx[1] for idx in idxs_going_to_each_key]
    atts_new = Dict(new_keys .=> collect(values(atts_integ))[idxs_new_atts]) #each label is associated to an attractor; that attractor, amongs potentially many that lead to the same labels, is the one that appears first in collect(values(atts_integ)) (a bit arbitrary but ok)
    return atts_new
end


function integrate_and_regroup_and_rematch(fam, prange, pidx, atts_info, fs_curves; kwargs...)
    fs_curves_new = deepcopy(fs_curves) 
    atts_info_new = deepcopy(atts_info)
       
    (; mapper, distance, threshold) = fam
    
    reset!(mapper)
    
    for idx in eachindex(prange)
        p = prange[idx]
        set_parameter!(mapper.ds, pidx, p)
        atts, fs = atts_info[idx], fs_curves[idx]
        atts_new, fs_new = integrate_and_regroup(mapper, atts, fs; kwargs...)
        atts_info_new[idx] = atts_new 
        fs_curves_new[idx] = fs_new 
    end

    match_continuation!(fs_curves_new, atts_info_new; distance, threshold)
    return atts_info_new, fs_curves_new
end