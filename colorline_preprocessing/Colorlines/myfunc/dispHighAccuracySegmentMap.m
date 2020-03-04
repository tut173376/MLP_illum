function dispHighAccuracySegmentMap(estLight_uv, lightNum, allCoef, scale, seg, im)

% estLight_uv : ����l
% dist_thresh : ����l�Ƃ̋��e����
dist_thresh = 0.05*scale; % ����l�܂ł̍ő勖�e����
highAccuracySegment = cell(lightNum, 1);
coefNum = size(allCoef, 1);

for lightID = 1:lightNum
    % est�����̂܂�, a,b��/scale�Ń����W�����킹�Ă���肭�����Ȃ���肪�c
    est = estLight_uv(lightID,:)*scale
    linesegs = [];

    for lineID = 1:coefNum
        a = allCoef(lineID, 1);%/scale;
        b = allCoef(lineID, 2);%/scale;
        fprintf('%f, %f\n', a, b);
        
        % �_(x0,y0)�ƒ���ax+by+c = 0�̋��� : |ax0+by0+c| / sqrt(a^2+b^2)
        dist = abs(a*est(1) - est(2) + b) /  sqrt(a^2 + 1);
        fprintf('line[%d] : %f/%f = %f\n', lineID, abs(a*est(1) - est(2) + b), sqrt(a^2 + 1), dist);
        
        if dist < dist_thresh
            crntDist = [allCoef(lineID, 5), dist];
            linesegs = [linesegs; crntDist];
        end
        
        % --- �\��
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

% �����̌����Ɍׂ��Ă���Z�O�����g������
distsA = highAccuracySegment{1};
distsB = highAccuracySegment{2};

% �d���v�f�𒊏o
[~, indexA, indexB] = intersect(distsA(:,1), distsB(:,1));

% �������ɋ߂������ɑ΂��鏈��
for id = 1:size(indexA, 1)
    dA = distsA(indexA(id), 2);
    dB = distsB(indexB(id), 2);
    
    % �Ɩ�A�̕����߂�
    if dA < dB
        %fprintf('lightA(%d):%f - lightB(%d):%f -> remove lightB\n', indexA(id), dA, indexB(id), dB);
        distsB(indexB(id), :) = [NaN,NaN];
    else % �Ɩ�B�̕����߂�
        %fprintf('lightA(%d):%f - lightB(%d):%f -> remove lightA\n', indexA(id), dA, indexB(id), dB);
        distsA(indexA(id), :) = [NaN,NaN];
    end
end
% NaN���܂ލs�̍폜
distsA(any(isnan(distsA),2),:) = [];
distsB(any(isnan(distsB),2),:) = [];
highAccuracySegment{1} = distsA;
highAccuracySegment{2} = distsB;


% �Ɩ����̕������w�����}�b�v�𐶐�
instImgList = cell(lightNum, 1);

for lightID = 1:lightNum
    % lightID�ɑΉ�����w�����}�b�v
    instructionMap = zeros(size(seg));
    
    % ���x�̍����Z�O�����g���X�g
    distList = highAccuracySegment{lightID};
    segList = distList(:,1);
    
    for id = 1:size(segList, 1)
        %fprintf('linesegs[%d]: %d\n', id, segList(id));
        tmpmap = (seg == segList(id));
        instructionMap = instructionMap | tmpmap;
    end
    % list�Ɋi�[
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