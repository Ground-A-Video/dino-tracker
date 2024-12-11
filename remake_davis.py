import pickle

num_frames = 24

path_to_pickle = 'tapvid/tapvid_davis_data_strided.pkl'
save_path = f'tapvid/tapvid_davis_data_strided_{num_frames}.pkl'

# Load the pickle file
benchmark_config = pickle.load(open(path_to_pickle, "rb"))
videos_data = benchmark_config['videos']    # 'list' data


for i in range(len(videos_data)):
    video_instance = benchmark_config['videos'][i]

    # truncate: 'target_points'
    del_keys = []
    for key,value in video_instance['target_points'].items():
        if key >= num_frames:
            del_keys.append(key)

    for del_key in del_keys:
        del video_instance['target_points'][del_key]

    for key, value in video_instance['target_points'].items():
        for i, value_item in enumerate(video_instance['target_points'][key]):
            video_instance['target_points'][key][i] = video_instance['target_points'][key][i][:num_frames]


    # truncate: 'occluded'
    del_keys = []
    for key,value in video_instance['occluded'].items():
        if key >= num_frames:
            del_keys.append(key)

    for del_key in del_keys:
        del video_instance['occluded'][del_key]

    for key, value in video_instance['occluded'].items():
        for i, value_item in enumerate(video_instance['occluded'][key]):
            video_instance['occluded'][key][i] = video_instance['occluded'][key][i][:num_frames]


    # truncate: 'is_traj_occluded'
    del_keys = []
    for key,value in video_instance['is_traj_occluded'].items():
        if key >= num_frames:
            del_keys.append(key)

    for del_key in del_keys:
        del video_instance['is_traj_occluded'][del_key]


    # truncate: 'occlusion_length'
    del_keys = []
    for key,value in video_instance['occlusion_length'].items():
        if key >= num_frames:
            del_keys.append(key)

    for del_key in del_keys:
        del video_instance['occlusion_length'][del_key]


    # truncate: 'is_fg'
    del_keys = []
    for key,value in video_instance['is_fg'].items():
        if key >= num_frames:
            del_keys.append(key)

    for del_key in del_keys:
        del video_instance['is_fg'][del_key]

    
    # truncate: 'fg_local_rel_area'
    del_keys = []
    for key,value in video_instance['fg_local_rel_area'].items():
        if key >= num_frames:
            del_keys.append(key)

    for del_key in del_keys:
        del video_instance['fg_local_rel_area'][del_key]


    # truncate: 'max_occlusion_length'
    del_keys = []
    for key,value in video_instance['max_occlusion_length'].items():
        if key >= num_frames:
            del_keys.append(key)

    for del_key in del_keys:
        del video_instance['max_occlusion_length'][del_key]


    # truncate: 'query_points'
    del_keys = []
    for key,value in video_instance['query_points'].items():
        if key >= num_frames:
            del_keys.append(key)

    for del_key in del_keys:
        del video_instance['query_points'][del_key]


# Save modified pickle
with open(save_path, 'wb') as f:
    pickle.dump(benchmark_config, f)

print(f"Modified pickle file saved at: {save_path}")