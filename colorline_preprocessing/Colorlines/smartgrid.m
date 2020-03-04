function smartgrid(step)
newplot

for i = 0:step:1
    for j = 0:step:1
        hold on
        plot3([0 1], [j j], [i, i], 'Color', [0.5 0.5 0.5], 'LineStyle', '-')
        plot3([j j], [0 1], [i, i], 'Color', [0.5 0.5 0.5], 'LineStyle', '-')
        plot3([i i], [j j], [0 1], 'Color', [0.5 0.5 0.5], 'LineStyle', '-')
        hold off
    end
end

axis([0 1 0 1 0 1])
axis square, grid on, view(-20,15);
ax = gca; ax.BoxStyle = 'full'; ax.Color = [0.9, 0.9, 0.9];
ax.XLabel.String = 'R'; ax.YLabel.String = 'G'; ax.ZLabel.String = 'B';
end