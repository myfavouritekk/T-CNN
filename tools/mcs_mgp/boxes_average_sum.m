function values = boxes_average_sum(map, boxes, box_ratio)
% Author Hongsheng Li

if nargin == 2
    box_ratio = 1.0;
end

[m, n] = size(map);
accum_map = cumsum(cumsum(map,1),2);

col1 = boxes(:,1);
row1 = boxes(:,2);
col2 = boxes(:,3);
row2 = boxes(:,4);

n_row = row2 - row1 + 1;
n_col = col2 - col1 + 1;

col1 = round(col1 + 0.5*(1-box_ratio)*n_col);
row1 = round(row1 + 0.5*(1-box_ratio)*n_row);
col2 = round(col2 - 0.5*(1-box_ratio)*n_col);
row2 = round(row2 - 0.5*(1-box_ratio)*n_row);

col1 = max(1, col1);
row1 = max(1, row1);
col2 = min(n, col2);
row2 = min(m, row2);

n_row = row2 - row1 + 1;
n_col = col2 - col1 + 1;

col1 = col1-1;
row1 = row1-1;
col_out_idx = col1==0;
row_out_idx = row1==0;
corner_out_idx = col_out_idx | row_out_idx;

col1(col_out_idx) = 1;
row1(row_out_idx) = 1;

sum_idx = sub2ind([m,n],row2,col2);
row_idx = sub2ind([m,n],row1,col2);
col_idx = sub2ind([m,n],row2,col1);
corner_idx = sub2ind([m,n],row1,col1);

sum_values = accum_map(sum_idx);
corner_values = accum_map(corner_idx);
col_values = accum_map(col_idx);
row_values = accum_map(row_idx);

corner_values(corner_out_idx) = 0;
col_values(col_out_idx) = 0;
row_values(row_out_idx) = 0;

values = sum_values - col_values - row_values + corner_values;
values = values ./ (n_row .* n_col);
