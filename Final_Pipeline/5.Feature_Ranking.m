load('features.mat'); % Assuming it contains a variable named 'features'
load('labels.mat');   % Assuming it contains a variable named 'labels'
num_channels = 22;
num_features = 17;
ranked_feature_indices = zeros(num_channels, num_features); % To store ranked indices
ranked_features = zeros(size(features)); % To store ranked features

for ch = 1:num_channels
    feature_matrix = squeeze(features(:, ch, :)); % Extract feature matrix for one channel (1148 x 17)
    
    % Rank features using rankfeatures
    idx = rankfeatures(feature_matrix', labels, 'Criterion', 'ttest'); 
    
    ranked_feature_indices(ch, :) = idx; % Store feature ranking order
    ranked_features(:, ch, :) = feature_matrix(:, idx); % Store ranked features
end

save('ranked_features.mat', 'ranked_features', 'ranked_feature_indices');