close all;

firstfig = figure;
seg2 = seg > 0;
for n = find(invalidSegIdx==true)' % invalidSegIdx�͏c�x�N�g��
    seg2(seg == n) = false;
end
[x,y] = getpts(firstfig, imshow(im.*repmat(seg2,[1 1 3])));
close(firstfig)

segID = seg(round(y(1)), round(x(1)))

% --- �אڂ��܂�
segregion = seg==segID;

se = strel('square',10);
expand = imdilate(segregion, se);
periphery = expand-segregion; % ����

neighborSegID = [];
[px, py] = find(periphery);
for id = 1:size(px,1)
    neighborSegID = [neighborSegID, seg(px(id), py(id))];
end
neighborSegID = unique(neighborSegID);
neighborSegID(neighborSegID==0) = [];
% 
% tseg = zeros(size(seg));
% for id = 1:size(neighborSegID, 2)
%     tmp = seg==neighborSegID(id);
%     %fprintf('>>%d\n', neighborSegID(id));
%     tseg = tseg|tmp;
% end
% ---

weight = 0;
%for lineID = 1:coefNum
% ���݂̃Z�O�����gID
currentSegID = segID;%allCoef(lineID, 5);
[cx,cy] = find(seg==currentSegID);
currentLightID = lightIDMap(cx(1),cy(1));

for id = 1:size(neighborSegID,2)
    [x,y] = find(seg==neighborSegID(id));
    neighborLightID = lightIDMap(x(1),y(1));
    
    % �����ƈႤ�Ɩ��������Ă���
    if currentLightID ~= neighborLightID
        weight = weight-1;
        fprintf('(%d) <-> nei(%d):%d\n', currentLightID, neighborSegID(id), neighborLightID);
    end
    if currentLightID == neighborLightID
        weight = weight+1;
        fprintf('(%d) <-> nei(%d):%d\n', currentLightID, neighborSegID(id), neighborLightID);
    end
end
if weight > 0
    fprintf('ok (%d)\n', weight);
else
    fprintf('gm (%d)\n', weight);
end

tseg = seg==segID; % �ΏۃZ�O�����g
nseg = ~tseg;%seg~=segID; % �Ώۂł͂Ȃ��Z�O�����g(�Â�����)

tmat = repmat(tseg,[1 1 3]);
nmat = repmat(nseg,[1 1 3])*0.5;

mat = nmat + tmat;
ttt = ones(size(seg, 1), size(seg, 2), 3);
cspixels = im.*mat.*single(perim8bit); % �摜����

% ���Ɍ��摜, �E��superpixel��\��
figure;
subplot(1,2,1), imshow(cspixels);
subplot(1,2,2), imshow(double(bg)/255);

allCoefIndex = find(allCoef(:,5)==segID);

hold on
plot( idx_u(validClusterIdx), idx_v(validClusterIdx), 'x', 'MarkerEdgeColor', 'w', 'MarkerSize', 10 )

seg_range = range(1):0.3:range(2);

seg_cl = allCoef(allCoefIndex, 1)*seg_range + allCoef(allCoefIndex, 2); % �Z�O�����g��colorLine
plot(seg_range, seg_cl, 'w-');
hold off

xlabel('u (Luv)','FontSize',20); ylabel('v (Luv)','FontSize',20);
axis on
