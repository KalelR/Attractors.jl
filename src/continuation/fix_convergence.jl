# export continuation_fix_convergence

using Attractors: match_attractor_ids!, swap_dict_keys!, set_parameter!, reset!, extract_feature, retract_keys_to_consecutive

function continuation_fix_convergence(fam, prange, pidx, fs_curves, atts_info; T=nothing, Ttr=nothing, Δt=nothing)
    fs_curves_new = deepcopy(fs_curves) 
    atts_info_new = deepcopy(atts_info)
       
    @show fam.mapper.ds.diffeq
    (; mapper, distance, threshold) = fam
    
    if isnothing(T) 
        Ttr = mapper.Ttr; Δt = mapper.Δt; T = mapper.total
    end
    
    reset!(mapper)
    
    for idx in eachindex(prange)
        p = prange[idx]
        set_parameter!(mapper.ds, pidx, p)
        
        atts = atts_info[idx]
        fs = fs_curves[idx]

        atts_integ = Dict(k => trajectory(mapper.ds, T, att[end]; Ttr, Δt)[1] for (k, att) in atts)
        ts = Ttr:Δt:Ttr+T
        features_integ = Dict(k => mapper.featurizer(att, ts) for (k, att) in atts_integ) 
        
        new_keys, tmap, idxs_going_to_each_key = transition(features_integ, mapper)
        fs_new = _build_fs(fs, new_keys, tmap)
        atts_new = _build_atts(new_keys, idxs_going_to_each_key, atts_integ)
        
        if idx > 1
            current_attractors = atts_new 
            prev_attractors = atts_info_new[idx-1]
            rmap = match_attractor_ids!(
                current_attractors, prev_attractors; distance, threshold
            )
            # @show rmap 
            # @show fs 
            # @show fs_new
            swap_dict_keys!(fs_new, rmap)
        end 
        atts_info_new[idx] = atts_new 
        fs_curves_new[idx] = fs_new
    end
    
    # Normalize to smaller available integers for user convenience
    rmap = retract_keys_to_consecutive(fs_curves_new)
    for (da, df) in zip(atts_info_new, fs_curves_new)
        swap_dict_keys!(da, rmap)
        swap_dict_keys!(df, rmap)
    end 

    return fs_curves_new, atts_info_new
end

function transition(features_integ, mapper)
    group_labels = group_features(collect(values(features_integ)), mapper.group_config)
    @show group_labels
    tmap = Dict(collect(keys(features_integ)) .=> group_labels)#map_keys_from_previous_grouping_to_keys_from_new_grouping
    new_keys = unique(group_labels) 
    idxs_going_to_each_key = map(ulab->findall(x->x==ulab, group_labels), new_keys)
    return new_keys, tmap, idxs_going_to_each_key
end

function _build_fs(fs::Dict{A,B}, new_keys, tmap) where {A,B}
    fs_new = Dict{A,B}()
    for key in new_keys 
        prev_keys_going_to_new_key = [k for (k,v) in tmap if v == key]
        fs_prev_keys = map(k->fs[k], prev_keys_going_to_new_key)
        fs_new[key] = sum(fs_prev_keys)
    end
    return fs_new 
end

function _build_atts(new_keys, idxs_going_to_each_key, atts_integ)
    idxs_new_atts = [idx[1] for idx in idxs_going_to_each_key]
    atts_new = Dict(new_keys .=> collect(values(atts_integ))[idxs_new_atts]) #each label is associated to an attractor; that attractor, amongs potentially many that lead to the same labels, is the one that appears first in collect(values(atts_integ)) (a bit arbitrary but ok)
    return atts_new
end


# 
# function transition_map(mapper, atts::Dict{A, B}) where {A, B}
#     metric = mapper.group_config.clust_distance_metric
#     Ttr = mapper.Ttr; Δt = mapper.Δt; T = mapper.total
#     ts = Ttr:Δt:Ttr+T
#     features_prev = Dict(k => mapper.featurizer(att, ts) for (k, att) in atts) 
#     features_integ = Dict(k => extract_feature(mapper.ds, att[end], mapper) for (k, att) in atts)
#     
#     group_labels = group_features(collect(values(features_integ)), mapper.group_config)
#     tmap = Dict(keys(features_integ) .=> group_labels)
#     
#     return tmap
# end

# function transition_map(mapper, atts::Dict{A, B}) where {A, B}
#     metric = mapper.group_config.clust_distance_metric
#     Ttr = mapper.Ttr; Δt = mapper.Δt; T = mapper.total
#     ts = Ttr:Δt:Ttr+T
#     features_prev = Dict(k => mapper.featurizer(att, ts) for (k, att) in atts) 
#     features_integ = Dict(k => extract_feature(mapper.ds, att[end], mapper) for (k, att) in atts)
#     
#     tmap = Dict{A, A}()
#     for (i, (k_prev, feature_prev)) in enumerate(features_prev)
#         dist_to_features_integ = Dict(k_integ => evaluate(metric, feature_integ, feature_prev) for (k_integ, feature_integ) in features_integ)
#         _, idx_closest = findmin(collect(values(dist_to_features_integ)))
#         k_closest = collect(keys(dist_to_features_integ))[idx_closest]
#         
#         if k_closest != k_prev
#             @info "Transition! Attractor $k_prev transitioned to $k_closest"
#         end
#        
#         tmap[k_prev] = k_closest
#     end
#     
#     return tmap
# end
# 
# function update_and_delete(fs::Dict{A,B}, tmap) where {A,B,C}
#     fs_new = Dict{A,B}() #initialize at zero!
#     # atts_new = Dict{A,C}()
#     @info "start update and delete"
#     for (k_prev, v) in fs 
#         k_new = tmap[k_prev]
#         if !haskey(fs_new, k_new) fs_new[k_new] = 0.0 end
#         fs_new[k_new] += fs[k_prev]
#         
#         # if !haskey(atts_new, k_new) atts_new[k_new] = atts[k_prev] end
#     end
#     return fs_new
# end

# function update_and_delete(d::Dict, atts, tmap)
#     d_new = deepcopy(d)
#     atts_new = deepcopy(atts)
#     @info "start update and delete"
#     @show d 
#     @show atts 
#     @show tmap
#     for (k_prev, v) in d 
#         k_new = tmap[k_prev]
#         @show k_new 
#         @show d
#         if k_new != k_prev 
#             @show k_prev
#             @show d_new
#             @show atts_new
#             d_new[k_new] += d[k_prev]
#             delete!(d_new, k_prev)
#             delete!(atts_new, k_prev)
#         end
#     end
#     return d_new, atts_new
# end