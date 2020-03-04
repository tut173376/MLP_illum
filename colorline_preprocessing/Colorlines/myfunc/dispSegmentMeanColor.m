function dispSegmentMeanColor(seg, segNum, satMap, lapMap, th_pixNum, L, u, v, scale, light_hsv, bg)

figure, imshow(double(bg)./255), hold on
xlabel('u (Luv)','FontSize',20); ylabel('v (Luv)','FontSize',20);
axis on

for n = 1:segNum % 枠の0は飛ばす
    segIdx = seg == n;
    idx = segIdx & ~satMap;
    
    % edgeが1画素でもあれば飛ばす，または，有効画素が50以下なら飛ばす
    if sum(sum(lapMap( segIdx ))) == 0 && sum(idx(:)) > th_pixNum
        seg_L = L( idx );
        seg_u = u( idx ) * scale; % u : [0,0.7] -> [0,200]
        seg_v = v( idx ) * scale; % v : [0,0.7] -> [0,200]

        plot(mean(seg_u), mean(seg_v), 'x')
    end
end

% philips hueの出力値をuv色度図上に描画
light_rgb = hsv2rgb(light_hsv);
light_xyz = rgb2xyz( light_rgb, 'WhitePoint', 'd65', 'ColorSpace', 'adobe-rgb-1998' );
light_u = 4*light_xyz(:,1) ./ ( light_xyz(:,1) + 15*light_xyz(:,2) + 3*light_xyz(:,3) );
light_v = 9*light_xyz(:,2) ./ ( light_xyz(:,1) + 15*light_xyz(:,2) + 3*light_xyz(:,3) );
plot( light_u*scale, light_v*scale, 'o', 'MarkerEdgeColor', 'w', 'MarkerSize', 10 );

end