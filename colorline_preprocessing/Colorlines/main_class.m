function main_class(imname,filepath)
warning('off','all')
addpath('myfunc'); addpath('./func_LSC/');
addpath('./func_colorEstimation/');
addpath('amanatidesWooAlgorithm');
addpath(genpath('drtoolbox'));
addpath('isomap');

%imnamef = imname;
%imname = '1000.JPG';
csvname = erase(imname,'.jpg');
imnamef = strcat(filepath, imname);
%disp(imname);
img = resizeImg(double(imread(imnamef))./255);

%csvfile
allsegcl = strcat('./csvfile\allsegCL\',csvname,'.csv');
enabledsegcl = strcat('./csvfile\segCL\',csvname,'.csv');
segmentationcsv = strcat('./csvfile\seg\',csvname,'.csv');
segimg = strcat('./csvfile/segimg/',csvname,'.png');
fidrr = fopen("csvfile/results.csv",'a');

%% param
lowLv = 0.2; highLv = 0.99; th_prc = 100;
th_pixNum = 200;
scale = 20; th_angle = 10; th_pca = 0.8; t_offset = 0.2;
lsc_ratio = 0.075; lsc_scale_factor = 50;
grid3D.nx = scale; grid3D.ny = scale; grid3D.nz = scale;
grid3D.minBound = [0, 0, 0]';
grid3D.maxBound = [1, 1, 1]';

line_length = 1.0; % 3D媒介変数

%% LSC
fft_img = fft2(img); fft_img = fftshift(fft_img);
mean_fft = mean(mean(mean(abs(fft_img))));
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

csvwrite(segmentationcsv,seg)
fid = fopen(allsegcl,'wt');
if fid>0
    for k=1:size(allCoef)
        if k ~= NanCell
            fprintf(fid,'%f,%f,%f,%f,%f,%f\n',allCoef{k,1},allCoef{k,2});
        else 
            fprintf(fid,'0.0,0.0,0.0,0.0,0.0,0.0\n');
        end
         
    end
    fclose(fid);
end

allCoef(NanCell,:) = []; 
fid = fopen(enabledsegcl,'wt');
if fid>0
    for k=1:size(allCoef)
       fprintf(fid,'%f,%f,%f,%f,%f,%f\n',allCoef{k,1},allCoef{k,2});
    end
    fclose(fid);
end

[x, y] = size(seg);
segtmp = ones(x,y);
for i = 1:x
    for j = 1:y
        for k = 1:size(NanCell)
            if NanCell(k)==seg(i,j)
               segtmp(i,j)=0;
            end
        end
     end
end

lscimg= img.*segtmp;
perimeter = seg == 0;
imwrite( imoverlay( lscimg, perimeter, 'black'), segimg )


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


voxel( voxel_start, voxel_size, illum, 0.5);

saveCurrentFigure(strcat('./csvfile/CL/',csvname,'_CL.png'));

fprintf(' > illum: (r,g,b)=(%.3f, %.3f, %.3f) \n', illum(1), illum(2), illum(3))
p = zeros(300,300,3); p(:,:,1) = illum(1); p(:,:,2) = illum(2); p(:,:,3) = illum(3);

c_img = img;
c_img(:,:,1) = img(:,:,1)/illum(1); c_img(:,:,2) = img(:,:,2)/illum(2); c_img(:,:,3) = img(:,:,3)/illum(3);
%figure, subplot(2,2,[1 3]), imshow(p), subplot(2,2,2), imshow(img), subplot(2,2,4), imshow(c_img)
imwrite( c_img, strcat('csvfile/results/', csvname ,'_correct.png') );
imwrite( p, strcat('csvfile/illum/', csvname ,'_illum.png') );
%imwrite( cat(2, img,c_img), strcat('outImgs/results.png') );
fprintf(fidrr,'%s,%f,%f,%f\n',imname, illum(1), illum(2), illum(3));
%{
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
%saveCurrentFigure('./outImgs/voting.png');

%% 投票の多い上位セルの可視化
candidateColors(voting, scale, 5)
 
%% 多様体学習の考えに従って次元圧縮をしてからクラスタリング
k = 2;

nzVotingInd = find(voting(:,:,:)~=0);
[ind1, ind2, ind3] = ind2sub([scale, scale, scale], nzVotingInd);
uni_allCP = [ind1, ind2, ind3];

no_dims = round(intrinsic_dim(uni_allCP, 'MLE')); % 適切な次元数(圧縮後はno_dims次元)
disp(no_dims);
if no_dims > size(uni_allCP,2) % 元の次元数を超えないようにする
    no_dims = size(uni_allCP,2);
end
[mappedX, mapping] = compute_mapping(uni_allCP, 'Isomap', no_dims); % Isomapにより次元圧縮
idx = kmeans(mappedX, k); % マッピングされた値に対してk-means


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
    imwrite( p, strcat('csvfile/illums/', csvname ,'_illum_clusters', num2str(n), '.png') );
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
end
%}
fclose('all');
end