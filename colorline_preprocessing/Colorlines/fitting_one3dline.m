rng(5,'twister');
seg_rgb = mvnrnd([0 0 0], [1 .2 .7; .2 1 0; .7 0 1],50);


[coeff,score,roots] = pca(seg_rgb);
[n,p] = size(seg_rgb);
meanX = mean(seg_rgb,1);

dirVect = coeff(:,1)
t = [min(score(:,1))-.2, max(score(:,1))+.2];
endpts = [meanX + t(1)*dirVect'; meanX + t(2)*dirVect'];
plot3(endpts(:,1),endpts(:,2),endpts(:,3),'k-');

Xfit1 = repmat(meanX,n,1) + score(:,1)*coeff(:,1)';
X1 = [seg_rgb(:,1) Xfit1(:,1) nan*ones(n,1)];
X2 = [seg_rgb(:,2) Xfit1(:,2) nan*ones(n,1)];
X3 = [seg_rgb(:,3) Xfit1(:,3) nan*ones(n,1)];
hold on
plot3(X1',X2',X3','b-', seg_rgb(:,1),seg_rgb(:,2),seg_rgb(:,3),'bo');
hold off
maxlim = max(abs(seg_rgb(:)))*1.1;
axis([-maxlim maxlim -maxlim maxlim -maxlim maxlim]);
axis square
view(-9,12);
grid on