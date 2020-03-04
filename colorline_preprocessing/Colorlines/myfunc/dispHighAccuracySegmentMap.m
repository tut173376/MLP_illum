function dispHighAccuracySegmentMap(estLight_uv, lightNum, allCoef, scale, seg, im)

% estLight_uv : 推定値
% dist_thresh : 推定値との許容距離
dist_thresh = 0.05*scale; % 推定値までの最大許容距離
highAccuracySegment = cell(lightNum, 1);
coefNum = size(allCoef, 1);

for lightID = 1:lightNum
    % estをそのまま, a,bを/scaleでレンジを合わせても上手くいかない問題が…
    est = estLight_uv(lightID,:)*scale
    linesegs = [];

    for lineID = 1:coefNum
        a = allCoef(lineID, 1);%/scale;
        b = allCoef(lineID, 2);%/scale;
        fprintf('%f, %f\n', a, b);
        
        % 点(x0,y0)と直線ax+by+c = 0の距離 : |ax0+by0+c| / sqrt(a^2+b^2)
        dist = abs(a*est(1) - est(2) + b) /  sqrt(a^2 + 1);
        fprintf('line[%d] : %f/%f = %f\n', lineID, abs(a*est(1) - est(2) + b), sqrt(a^2 + 1), dist);
        
        if dist < dist_thresh
            crntDist = [allCoef(lineID, 5), dist];
            linesegs = [linesegs; crntDist];
        end
        
        % --- 表示
%         figure(5)
%         imshow(double(bg)/255);
%         hold on
%         plot( est(1), est(2), '.', 'MarkerEdgeColor', 'w', 'MarkerSize', 10 )
%         plotX = round( range(1):range(2) );
%         if dist < dist_thresh
%             plot( plotX,allCoef(lineID, 1)*plotX+allCoef(lineID, 2), 'Color', 'w' );
%         else
%             plot( plotX,allCoef(lineID, 1)*plotX+allCoef(lineID, 2), 'Color', 'r' );
%         end
%         text('units','pixels','position',[100 150],'fontsize',12,'Color', [1 1 0.8], 'string',strcat('dist=', num2str(dist)));
%         hold off
%         pause;
        % ---
    end
    highAccuracySegment{lightID} = linesegs;
end

% 複数の光源に跨っているセグメントを除去
distsA = highAccuracySegment{1};
distsB = highAccuracySegment{2};

% 重複要素を抽出
[~, indexA, indexB] = intersect(distsA(:,1), distsB(:,1));

% 両光源に近い直線に対する処理
for id = 1:size(indexA, 1)
    dA = distsA(indexA(id), 2);
    dB = distsB(indexB(id), 2);
    
    % 照明Aの方が近い
    if dA < dB
        %fprintf('lightA(%d):%f - lightB(%d):%f -> remove lightB\n', indexA(id), dA, indexB(id), dB);
        distsB(indexB(id), :) = [NaN,NaN];
    else % 照明Bの方が近い
        %fprintf('lightA(%d):%f - lightB(%d):%f -> remove lightA\n', indexA(id), dA, indexB(id), dB);
        distsA(indexA(id), :) = [NaN,NaN];
    end
end
% NaNを含む行の削除
distsA(any(isnan(distsA),2),:) = [];
distsB(any(isnan(distsB),2),:) = [];
highAccuracySegment{1} = distsA;
highAccuracySegment{2} = distsB;


% 照明数の分だけ指示線マップを生成
instImgList = cell(lightNum, 1);

for lightID = 1:lightNum
    % lightIDに対応する指示線マップ
    instructionMap = zeros(size(seg));
    
    % 精度の高いセグメントリスト
    distList = highAccuracySegment{lightID};
    segList = distList(:,1);
    
    for id = 1:size(segList, 1)
        %fprintf('linesegs[%d]: %d\n', id, segList(id));
        tmpmap = (seg == segList(id));
        instructionMap = instructionMap | tmpmap;
    end
    % listに格納
    instImgList{lightID} = im.*repmat(instructionMap,[1 1 3]);
end

figure;

% u'v'l -> XYZ
upvpl2xyz = makecform('upvpl2xyz');

L1color = estLight_uv(1,:); L1color(3) = 0.75;
L1xyz = applycform(L1color, upvpl2xyz);
L1rgb = xyz2rgb(L1xyz, 'WhitePoint', 'D65');
L1pix = zeros(100,100,3); L1pix(:,:,1)=L1rgb(1); L1pix(:,:,2)=L1rgb(2); L1pix(:,:,3)=L1rgb(3);
subplot(2,2,1), imshow(L1pix);
L2color = estLight_uv(2,:); L2color(3) = 0.75;
L2xyz = applycform(L2color, upvpl2xyz);
L2rgb = xyz2rgb(L2xyz, 'WhitePoint', 'D65');
L2pix = zeros(100,100,3); L2pix(:,:,1)=L2rgb(1); L2pix(:,:,2)=L2rgb(2); L2pix(:,:,3)=L2rgb(3);
subplot(2,2,2), imshow(L2pix);

subplot(2,2,3);imshow( instImgList{1} );
subplot(2,2,4);imshow( instImgList{2} );