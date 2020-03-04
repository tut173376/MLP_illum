clc, close all, clear all
warning('off','all')

addpath('myfunc'); addpath('./func_LSC/');
addpath('./func_colorEstimation/');
addpath('amanatidesWooAlgorithm');
addpath(genpath('drtoolbox'));
addpath('isomap');

% references
% disp voxels
% http://jp.mathworks.com/matlabcentral/fileexchange/46564-voxel-m
% raycasting
% http://jp.mathworks.com/matlabcentral/fileexchange/26852-a-fast-voxel-traversal-algorithm-for-ray-tracing
% calculate 3d vector
% https://jp.mathworks.com/help/stats/examples/fitting-an-orthogonal-regression-using-principal-components-analysis.html

imname = '8D5U5525';
filepath = strcat('X:\参考論文\IlluminantEstimation\Datasets\Color-checker\myJPG\', imname, '.JPG');
img = resizeImg(double(imread(filepath))./255);
% img = resizeImg(double(imread('X:\testImgs\sip2017\scene1\IMG_0005.JPG'))./255);

%% param
lowLv = 0.2; highLv = 0.99; th_prc = 100;
th_pixNum = 200;
scale = 20; th_angle = 10; th_pca = 0.8; t_offset = 0.2;
lsc_ratio = 0.075; lsc_scale_factor = 50;
grid3D.nx = scale; grid3D.ny = scale; grid3D.nz = scale;
grid3D.minBound = [0, 0, 0]';
grid3D.maxBound = [1, 1, 1]';

%% memo
fid = fopen('./outImgs/preferences.txt', 'w');
fprintf(fid, '%s\n', datetime('now','InputFormat','uuuu-MM-dd''T''HH:mmXXX','TimeZone','local'));
fprintf(fid, 'lowLv = %f;\n', lowLv);
fprintf(fid, 'highLv = %f;\n', highLv);
fprintf(fid, 'th_prc = %d;\n', th_prc);
fprintf(fid, 'th_pixNum = %d;\n', th_pixNum);
fprintf(fid, 'scale = %d;\n', scale);
fprintf(fid, 'th_angle = %d;\n', th_angle);
fprintf(fid, 'th_pca = %f;\n', th_pca);
fprintf(fid, 't_offset = %f;\n', t_offset);
fprintf(fid, 'lsc_ratio = %d;\n', lsc_ratio);
fprintf(fid, 'lsc_scale_factor = %d;\n', lsc_scale_factor);
fclose(fid);

line_length = 1.0; % 3D媒介変数

%% LSC
fft_img = fft2(img); fft_img = fftshift(fft_img);
mean_fft = mean(mean(mean(abs(fft_img))));
imwrite( cat(2, img, fft_img./100), strcat('outImgs/fft_ave[', num2str(mean_fft, '%.3f'), ']_sp[', num2str(round(mean_fft)*lsc_scale_factor), '].png') );
seg = lsc_superpixels(img, round(mean_fft)*lsc_scale_factor, lsc_ratio);

%% map
r = img(:,:,1); g = img(:,:,2); b = img(:,:,3);
lupvp = rgb2lupvp(double(img)); L = lupvp(:,:,1); u = lupvp(:,:,2); v = lupvp(:,:,3);
[lapMap, satMap] = makeLSMap(L, img, lowLv, highLv, th_prc);

smartgrid(0.2)
allCoef = cell(max(max(seg)), 2);
allCP = [];

%% Colorlines
for n = 1:max(max(seg)) % 枠の0は飛ばす
    segIdx = seg == n;
    idx = segIdx & ~satMap;
    
    if sum(sum(lapMap( segIdx ))) == 0 && sum(idx(:)) > th_pixNum
        seg_r = r( idx ); seg_g = g( idx ); seg_b = b( idx );
        seg_rgb = [seg_r(:), seg_g(:), seg_b(:)];
        
        %% calculate colorline vector
        [coeff,score,roots] = pca(seg_rgb);
        [l,~] = size(seg_rgb);
        mean_rgb = mean(seg_rgb,1);
        pctExplained = roots' ./ sum(roots);
        
        if pctExplained(1) > 0.8
            % line
            dirVector = coeff(:,1);
            t = [-line_length line_length];%[min(score(:,1)), max(score(:,1))];
            endpts = [mean_rgb + t(1)*dirVector'; mean_rgb + t(2)*dirVector'];
            hold on
            plot3(endpts(:,1),endpts(:,2),endpts(:,3),'-');
            allCoef{n,1} = dirVector;
            allCoef{n,2} = mean_rgb;
        end
    end
end

NanCell = find( cellfun(@isempty,allCoef(:,1)) ); % 空のセルを探索
allCoef(NanCell,:) = []; % 空のセルを除去

%% 交点計算
grid3D.nx = scale; grid3D.ny = scale; grid3D.nz = scale;
grid3D.minBound = [0, 0, 0]';
grid3D.maxBound = [1, 1, 1]';
voting = zeros(scale, scale, scale);

boxels = cell(size(allCoef,1),1);

tic
for n = 1:size(allCoef,1)
    vect1 = allCoef{n,1}; point1 = allCoef{n,2};
    boxel1 = zeros(scale, scale, scale);    
    origin = [point1(1), point1(2), point1(3)]';
    direction = [vect1(1), vect1(2) vect1(3)]';
    t = t_offset; origin = origin + t*direction; % Colorlineを基準点の両側で考える

    ivs = amanatidesWooAlgorithm(origin, direction, grid3D, 0);
    if find( cellfun(@isempty,ivs) ) % Colorlineとvoxelが1つも交差しない
        boxels{n} = boxel1;
        continue;
    end
    for lp = 1:size(ivs,2)
        sp = int8(ivs{lp});
        boxel1( sp(1), sp(2), sp(3) ) = 1;
    end
    boxels{n} = boxel1;
end

%% line1
for n = 1:size(allCoef,1)
    boxel1 = boxels{n};
    vect1 = allCoef{n,1}; point1 = allCoef{n,2};
    
    %% line2
    for m = n:size(allCoef,1)
        vect2 = allCoef{m,1}; point2 = allCoef{m,2};   

        angle = acosd( (vect1(1)*vect2(1) + vect1(2)*vect2(2) + vect1(3)*vect2(3)) / (norm(vect1)*norm(vect2)) );
        if angle > th_angle
            boxel2 = boxels{m};        
            crosspoints = boxel1&boxel2;
            voting = voting+crosspoints;
        end
    end
end
toc

[bm, bi] = max(max(max(voting(:,:,:))));
[gm, gi] = max(max(voting(:,:,bi)));
[rm, ri] = max(voting(:,gi,bi));

voxel_start = [(ri-1)/scale (gi-1)/scale (bi-1)/scale];
voxel_size = [1.0/scale 1.0/scale 1.0/scale];
% if size(allCP) == 0
%     illum = [1.0 1.0 1.0];
% else
    illum = [ri/scale gi/scale bi/scale];
% end

% %% クラスタリングして平均が灰色に近い方を採用
% k = 2; idx = kmeans(allCP, k);
% reliability = zeros(k,1);
% for n = 1:k
%     clusterIndex = find(idx==n); votes = allCP(clusterIndex,:);
%     [C_CP,~,icp] = unique( votes, 'rows' );
%     mode_icp = mode(icp);
%     est_illums{n} = C_CP(mode_icp,:)/scale;
% 
%     gw_img = img;
%     gw_img(:,:,1) = gw_img(:,:,1)./est_illums{n}(1);
%     gw_img(:,:,2) = gw_img(:,:,2)./est_illums{n}(2);
%     gw_img(:,:,3) = gw_img(:,:,3)./est_illums{n}(3);
%     ave_gw_img = [mean(mean(gw_img(:,:,1))), mean(mean(gw_img(:,:,2))), mean(mean(gw_img(:,:,3)))];
%     gray = [0.5, 0.5, 0.5];
%     reliability(n) = acosd( dot(ave_gw_img, gray)/(norm(ave_gw_img)*norm(gray)) );
% end
% 
% [~, minid] = min(reliability);
% illum = est_illums{minid};

voxel( voxel_start, voxel_size, illum, 0.5);

saveCurrentFigure('./outImgs/Colorlines.png');

fprintf(' > illum: (r,g,b)=(%.3f, %.3f, %.3f) \n', illum(1), illum(2), illum(3))
p = zeros(300,300,3); p(:,:,1) = illum(1); p(:,:,2) = illum(2); p(:,:,3) = illum(3);

c_img = img;
c_img(:,:,1) = img(:,:,1)/illum(1); c_img(:,:,2) = img(:,:,2)/illum(2); c_img(:,:,3) = img(:,:,3)/illum(3);
figure, subplot(2,2,[1 3]), imshow(p), subplot(2,2,2), imshow(img), subplot(2,2,4), imshow(c_img)
imwrite( img, strcat('outImgs/original.png') );
imwrite( c_img, strcat('outImgs/correct.png') );
imwrite( p, strcat('outImgs/illum.png') );
imwrite( cat(2, img,c_img), strcat('outImgs/results.png') );

perimeter = seg==0;
imwrite( imoverlay( img, perimeter, 'black'),strcat('outImgs/lsc(frame).png') )

% gt_illum = Copy_of_getPatchMeanColor(filepath, strcat('X:\参考論文\IlluminantEstimation\Datasets\Color-checker\ColorCheckerDatabase_MaskCoordinates\coordinates\', imname, '_macbeth.txt'), strcat('outImgs//chart_cripped.png'));
% angular_error = acosd( dot(illum, gt_illum)/(norm(illum)*norm(gt_illum)) );
% 
% gt_p = zeros(300,300,3); gt_p(:,:,1) = gt_illum(1); gt_p(:,:,2) = gt_illum(2); gt_p(:,:,3) = gt_illum(3);
% imwrite( cat(2, p, gt_p), strcat('outImgs/results_ae', num2str(angular_error, '%.2f'), '[deg].png') );

%% vis voting result(voxels color)
figure, smartgrid(0.1)
for x = 1:scale
    for y = 1:scale
        for z = 1:scale
            if voting(x,y,z) ~= 0
                voxel_start = [x-1 y-1 z-1]/scale;
                voxel_size = [1.0/scale 1.0/scale 1.0/scale];
                color = [x y z]/scale;
                voxel( voxel_start, voxel_size, color, voting(x,y,z)/rm);
            end
        end
    end
end
saveCurrentFigure('./outImgs/voting.png');

%% 投票の多い上位セルの可視化
candidateColors(voting, scale, 5)
 
%% 多様体学習の考えに従って次元圧縮をしてからクラスタリング
k = 2;

nzVotingInd = find(voting(:,:,:)~=0);
[ind1, ind2, ind3] = ind2sub([scale, scale, scale], nzVotingInd);
uni_allCP = [ind1, ind2, ind3];

% [uni_allCP, ia, ic] = unique(allCP, 'rows');
no_dims = round(intrinsic_dim(uni_allCP, 'MLE')); % 適切な次元数(圧縮後はno_dims次元)
if no_dims > size(uni_allCP,2) % 元の次元数を超えないようにする
    no_dims = size(uni_allCP,2);
end
[mappedX, mapping] = compute_mapping(uni_allCP, 'Isomap', no_dims); % Isomapにより次元圧縮
% [mappedX, mapping] = compute_mapping(uni_allCP, 'LLE', no_dims); % Locally Linear Embeddingにより次元圧縮
% [mappedX, mapping] = compute_mapping(uni_allCP, 'tSNE', no_dims); % t-Distributed Stochastic Neighbor Embeddingにより次元圧縮
idx = kmeans(mappedX, k); % マッピングされた値に対してk-means

% % mapping後のvoxelを可視化
% minC1 = min(mappedX(:,1))/scale; maxC1 = max(mappedX(:,1))/scale;
% minC2 = min(mappedX(:,2))/scale; maxC2 = max(mappedX(:,2))/scale;
% minC3 = min(mappedX(:,3))/scale; maxC3 = max(mappedX(:,3))/scale;
% figure
% for n = 1:size(mappedX, 1)
%     voxel_start = [mappedX(n,1), mappedX(n,2), mappedX(n,3)]/scale;
%     voxel_size = [maxC1-minC1 maxC2-minC2 maxC3-minC3]/scale;
%     color = [uni_allCP(n,1), uni_allCP(n,2), uni_allCP(n,3)]/scale;
%     voxel( voxel_start, voxel_size, color, voting(uni_allCP(n,1),uni_allCP(n,2),uni_allCP(n,3))/rm);
% end
% grid on, grid minor, axis square, grid on, view(-20,15);
% ax = gca; ax.BoxStyle = 'full'; ax.Color = [0.9, 0.9, 0.9];
% % axis([minC1 maxC1 minC2 maxC2 minC3 maxC3])
% saveCurrentFigure('./outImgs/Mapped.png');


% 各クラスで最も表が多いセルを推定色とする
for n = 1:k
    clusterIndex = find(idx==n); % クラスタnに属する要素のインデックス
    
    maxCnt = 0; maxInd = 0;
    for m = 1:size(clusterIndex,1)
        % ic: uni_allCPのインデックス
        % 次の行はクラスタnにおける要素mが, allCPにおいて何回出てきたか数えている
        vtInd = uni_allCP(clusterIndex(m),:);
        cnt = voting(vtInd(1),vtInd(2),vtInd(3));
        %cnt = size( find(ic==clusterIndex(m)), 1); %上2行と等価
        
        if cnt > maxCnt
            maxCnt = cnt;
            maxInd = m;
        end
    end
    % つまり, クラスタnで最も投票数が多いvoxelを抽出している
    % ここでのclusterIndex(maxInd)は, uni_allCP内で最も出現したrgb値を持つic
    indexes{n} = uni_allCP(clusterIndex(maxInd),:);
    
    p = zeros(500,500,3); p(:,:,1) = indexes{n}(1)/scale; p(:,:,2) = indexes{n}(2)/scale; p(:,:,3) = indexes{n}(3)/scale;
    imwrite( p, strcat('outImgs/illum_clusters', num2str(n), '.png') );
    fprintf('> class%d: (%.1f,%.1f,%.1f)\n', n, indexes{n}(1)/scale, indexes{n}(2)/scale, indexes{n}(3)/scale);
end

%% 各クラスで最も票が多いセル
figure, smartgrid(0.1)
for n = 1:k
    index = indexes{n};
    voxel_start = [index(1)-1 index(2)-1 index(3)-1]/scale;
    voxel_size = [1.0/scale 1.0/scale 1.0/scale];
    color = index/scale;
    voxel( voxel_start, voxel_size, color, voting(index(1), index(2), index(3))/rm);
end
saveCurrentFigure('./outImgs/clusterMaxCell.png');

%% クラスタの可視化
for n = 1:k
    clusterIndex = find(idx==n);
    colors = uni_allCP(clusterIndex,:);
    figure, smartgrid(0.1)

    for id = 1:size(colors, 1)
        voxel_start = [colors(id, 1)-1 colors(id, 2)-1 colors(id, 3)-1]/scale;
        voxel_size = [1.0/scale 1.0/scale 1.0/scale];
        color = [colors(id,1), colors(id,2), colors(id,3)]/scale;
        voxel( voxel_start, voxel_size, color, voting(colors(id,1), colors(id,2), colors(id,3))/rm);
    end
    saveCurrentFigure(strcat('./outImgs/clusters_', num2str(n), '.png'));
end

