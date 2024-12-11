import pickle

num_frames = 24
path_to_pickle = 'tapvid/tapvid_BADJA_data.pkl'
save_path = f'tapvid/tapvid_BADJA_data_{num_frames}.pkl'

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


    # truncate: 'segmentations'
    video_instance['segmentations'] = video_instance['segmentations'][:num_frames, ...]


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