function luminancePlot(seg_u, seg_L)

padding = 0.05;
plot(seg_u, seg_L, '.');
xlabel('seg_u'); ylabel('seg_L');
axis([min(seg_u)-padding, max(seg_u)+padding, min(seg_L)-padding, max(seg_L)+padding]);
xlabel('u (Luv)','FontSize',20); ylabel('L (Luv)','FontSize',20);
lsline;

end