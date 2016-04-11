function mcs_mgp(flow_root, score_root, output_root)
%% propagating still-image detection according to optical flows
% suppressing low-confidence classes accoridng to context info
%
% First written by Hongsheng Li. Refactored by Kai Kang.

% parameters
temporal_window_size = 7;
half_tws = floor(temporal_window_size / 2);
video_top_ratio = 0.0003;
top_bonus = 0.4;

% paths
time_step = 1;
output_root = fullfile(output_root,...
    sprintf('window_size_%d_time_step_%d_top_ratio_%f_top_bonus_%f_optflow',...
        temporal_window_size, time_step, video_top_ratio, top_bonus));

% mkdir_if_missing(output_root);
if ~exist(output_root, 'dir')
    mkdir(output_root);
end

video_name = dir(score_root);
video_name = video_name(3:end);
n_video = length(video_name);

for video_idx = 1:n_video
    %% optical flow
    if ~isdir(fullfile(score_root, video_name(video_idx).name))
        continue;
    end
    frame_name = dir(fullfile(score_root, video_name(video_idx).name, '*.mat'));
    n_frame = length(frame_name);
    fprintf('%d of %d total videos, name: %s.',...
        video_idx, n_video, video_name(video_idx).name);

    frame = struct('boxes',[],'zs',[]);
    frame(n_frame).boxes = [];
    neighbor_frame = frame;

    fprintf(' Loading boxes.');
    for frame_idx = 1:n_frame
        file_name = fullfile(score_root, video_name(video_idx).name, frame_name(frame_idx).name);
        dot_pos = findstr(frame_name(frame_idx).name, '.');
        dot_pos = dot_pos(1);
        optflow_name = fullfile(flow_root, video_name(video_idx).name, [frame_name(frame_idx).name(1:dot_pos-1) '.png']);

        frame(frame_idx) = load(file_name);
        if isempty(frame(frame_idx).boxes)
            continue;
            frame(frame_idx).boxes = zeros(0,4);
            frame(frame_idx).zs = zeros(0,30);
        end
        optflow = imread(optflow_name);
        x_map = single(optflow(:,:,1)) / 255 * 30 - 15;
        y_map = single(optflow(:,:,2)) / 255 * 30 - 15;
        [m,n] = size(x_map);
        box_avg_x = boxes_average_sum(x_map, frame(frame_idx).boxes);
        box_avg_y = boxes_average_sum(y_map, frame(frame_idx).boxes);

        for offset_idx = [-half_tws:-1 1:half_tws]
            neighbor_frame_idx = frame_idx + offset_idx;
            if neighbor_frame_idx < 1 || neighbor_frame_idx > n_frame
                continue;
            end

            boxes = frame(frame_idx).boxes;
            zs = frame(frame_idx).zs;
            if abs(offset_idx) >= 3
                zs(:,1) = -1e+5;
            end
            boxes = boxes + ([box_avg_x, box_avg_y, box_avg_x, box_avg_y] * offset_idx);
            boxes(:,1) = max(boxes(:,1),1);
            boxes(:,2) = max(boxes(:,2),1);
            boxes(:,3) = min(boxes(:,3),n);
            boxes(:,4) = min(boxes(:,4),m);
            neighbor_frame(neighbor_frame_idx).boxes = cat(1, neighbor_frame(neighbor_frame_idx).boxes, boxes);
            neighbor_frame(neighbor_frame_idx).zs = cat(1, neighbor_frame(neighbor_frame_idx).zs, zs);
        end
    end

    %% context
    all_zs = cat(1, frame(1:end).zs);
    n_box = size(all_zs,1);
    all_class_idx = repmat(1:30,n_box,1);
    all_zs = all_zs(:);
    all_class_idx = all_class_idx(:);
    [sorted_all_zs,sort_idx] = sort(all_zs, 'descend');
    sorted_class_idx = all_class_idx(sort_idx);
    n_top = round(n_box*video_top_ratio);
    if n_top == 0
       n_top = 1;
    end
    top_classes = unique(sorted_class_idx(1:n_top));
    fprintf(1, ' n_top = %d.', n_top);

    fprintf(1, ' Saving boxes.\n');
    for frame_idx = 1:n_frame
        clear boxes zs;
        boxes = cat(1, neighbor_frame(frame_idx).boxes, frame(frame_idx).boxes);
        zs = cat(1, neighbor_frame(frame_idx).zs, frame(frame_idx).zs);
        if ~isempty(zs)
            zs(:,top_classes) = zs(:,top_classes) + top_bonus;
        end

        output_dir = fullfile(output_root, video_name(video_idx).name);
        mkdir_if_missing(output_dir);
        output_path = fullfile(output_dir, frame_name(frame_idx).name);
        save(output_path, 'boxes', 'zs');
    end
    clear frame neighbor_frame;
end
