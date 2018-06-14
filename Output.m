%%
%Output_lambda
slice = 40;
imagesc(ldash(:,:,slice)');
colormap default
axis equal tight off
caxis ([0,0.0001])

%%
%Output
slice = 1;
subplot(1,2,1);
imagesc(E1(:,:,slice)');
colormap gray
axis equal tight off
caxis([0,0.5])

subplot(1,2,2);
imagesc(Out(:,:,slice)');
colormap gray
axis equal tight off
caxis([0,0.5])

%%
%Save
save_raw(Out,[InputPath CasePath{1,:} '_Output' '.raw'],'*single');
