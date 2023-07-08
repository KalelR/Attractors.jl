export GroupViaComparing

struct GroupViaComparing{R<:Union{Real, String}, M} <: GroupingConfig
    clust_distance_metric::M
    rescale_features::Bool #can remove prob
    optimal_radius_method::R
    use_mmap::Bool
end

function GroupViaComparing(;
    clust_distance_metric=Euclidean(), rescale_features=false, 
    optimal_radius_method::Union{Real, String} = "silhouettes_optim",
    use_mmap = false,
)
return GroupViaComparing(
    clust_distance_metric, rescale_features,
    optimal_radius_method, use_mmap,
)
end

function group_features(
    features, config::GroupViaComparing;
    kwargs...
)
    # ϵ_optimal = _extract_ϵ_optimal(features, config)
    ϵ_optimal = config.optimal_radius_method
    labels = _cluster_features_into_labels_comparing(features, config, ϵ_optimal; kwargs...)
    return labels
end

#TODO: check if calculating the distances on the fly is better than pre-calculating the distances; of course it allocates way less, which for lots of ics is prob better, but what for a few ics?
#distance_threshold is the maximum distance two points can be to be considered in the same cluster; above it they are different; so it just needs to be large enough to differentiate clusters
function _cluster_features_into_labels_comparing(features, config, distance_threshold; par_weight::Real = 0, plength::Int = 1, spp::Int = 1)
    labels_features = Vector{Int64}(undef, length(features)) #labels of all features
    labels_features[1] = 1
    clusters_idxs = [1] #idxs of the features that define each cluster
    cluster_labels = [1] #labels for the clusters, going from 1 : num_clusters
    next_cluster_label = 2
    metric = config.clust_distance_metric
    for idx_feature = 2:length(features)
        feature = features[idx_feature]
        dist_to_clusters = Dict(cluster_labels[idx] => evaluate(metric, feature, features[clusters_idxs[idx]]) for idx in eachindex(cluster_labels))
        dist_vals = collect(values(dist_to_clusters))
        min_dist, idx_min_dist = findmin(dist_vals)
        
        if min_dist > distance_threshold #bigger than threshold => new attractor
            push!(clusters_idxs, idx_feature)
            push!(cluster_labels, next_cluster_label)
            feature_label = next_cluster_label
            next_cluster_label += 1
        else #smaller => assign to closest cluster 
            idx_closest_cluster = collect(keys(dist_to_clusters))[idx_min_dist]
            feature_label = idx_closest_cluster
        end
        
        labels_features[idx_feature] = feature_label 
    end 
    return labels_features
end
# #TODO: check if calculating the distances on the fly is better than pre-calculating the distances; of course it allocates way less, which for lots of ics is prob better, but what for a few ics?
# #distance_threshold is the maximum distance two points can be to be considered in the same cluster; above it they are different
# function _cluster_features_into_labels_comparing(features, config, distance_threshold; par_weight::Real = 0, plength::Int = 1, spp::Int = 1)
#     labels_features = Vector{Int64}(undef, length(features)) #labels of all features
#     labels_features[1] = 1
#     clusters_idxs = [1] #idxs of the features that define each cluster
#     cluster_labels = [1] #labels for the clusters, going from 1 : num_clusters
#     next_cluster_label = 2
#     metric = config.clust_distance_metric
#     for idx_feature = 2:length(features)
#         feature = features[idx_feature]
#         dist_to_clusters = Dict(cluster_labels[idx] => evaluate(metric, feature, features[clusters_idxs[idx]]) for idx in eachindex(cluster_labels))
#         min_dist, idx_min_dist = findmin(values(dist_to_clusters)) 
#         idx_closest_cluster = keys(dist_to_clusters)[idx_min_dist]
#         labels_possible_clusters = [k for (k, v) in dist_to_clusters if v < distance_threshold]
#         
#         if length(labels_possible_clusters) == 0 #not close to any cluster, either too small radius or a new cluster! 
#             push!(clusters_idxs, idx_feature)
#             push!(cluster_labels, next_cluster_label)
#             feature_label = next_cluster_label
#             next_cluster_label += 1
#             # @info "new cluster! label = $feature_label, dist is $dist_to_clusters"
#             # @show features[clusters_idxs[cluster_labels]] 
#             # @show feature 
#         elseif length(labels_possible_clusters) == 1 #found the closest cluster!
#             feature_label = labels_possible_clusters[1]
#         else 
#             # possible_distances = [v for (k, v) in dist_to_clusters if v < distance_threshold]
#             idx_min = argmin(values(dist_to_clusters))
#             label_closest = keys(dist_to_clusters)[idx_min]
#             @warn "more than one cluster found for radius = $distance_threshold in idx $idx_feature. Should probably reduce the radius. Taking the closest: $label_closest." #TODO: implement this
#             @show features[clusters_idxs]
#             @show feature
#             @show dist_to_clusters
#             feature_label = label_closest 
#         end
#         
#         labels_features[idx_feature] = feature_label 
#     end 
#     return labels_features
# end