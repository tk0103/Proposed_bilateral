%load_path
fileID = fopen('InputPath.txt');
C = textscan(fileID,'%s');
fclose(fileID);
InputPath = C{1,1}; 
InputPath = cell2mat(InputPath);

fileID = fopen('CasePath.txt');
C = textscan(fileID,'%s');
fclose(fileID);
CasePath = C{1,1};

%load_data
E1 = load_raw([InputPath CasePath{1,:} '.raw'],'*single');
E2 = load_raw([InputPath CasePath{2,:} '.raw'],'*single');
E3 = load_raw([InputPath CasePath{3,:} '.raw'],'*single');
E4 = load_raw([InputPath CasePath{4,:} '.raw'],'*single');

siz = [436,436,63];
E1 = reshape(E1,siz);
E2 = reshape(E2,siz);
E3 = reshape(E3,siz);  
E4 = reshape(E4,siz);

%%
%confirm
slice = 30;
imagesc(E1(:,:,slice)');
axis tight equal off
colormap gray
caxis([0 0.6])