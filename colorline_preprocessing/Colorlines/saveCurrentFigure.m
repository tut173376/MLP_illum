function saveCurrentFigure(filename)

fig = gcf; fig.InvertHardcopy = 'off';
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 10 10];
saveas(gcf, filename);

end