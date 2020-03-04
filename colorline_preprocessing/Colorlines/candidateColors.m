function candidateColors(voting, scale, order)
if nargin == 1
    order = 5; % è„à 5åè
end

size = 500;
patches = zeros(size, size*order, 3);

for id = 1:order
    [~, bi] = max(max(max(voting(:,:,:))));
    [~, gi] = max(max(voting(:,:,bi)));
    [~, ri] = max(voting(:,gi,bi));
    
    illum = [ri/scale gi/scale bi/scale];
    p = zeros(size, size, 3); p(:,:,1) = illum(1); p(:,:,2) = illum(2); p(:,:,3) = illum(3);
    patches(:, 1+size*(id-1):size*id, : ) = p;
    
    voting(ri, gi, bi) = 0;
end
imwrite(patches, 'outImgs/candidates.png');

end