%%

img1 = load_raw('C:\Users\yourb\Desktop\proposed_bilateral_E3.raw','*single');
img2 = load_raw('C:\Users\yourb\Desktop\E3conventinal_bilateral.raw','*single');

siz = [480,480,80];
img1 = reshape(img1,siz);
imagesc(img1(:,:,30)');
colormap gray
axis equal tight off
figure;
img2 = reshape(img2,siz);
imagesc(img2(:,:,30)');
colormap gray
axis equal tight off
figure;

diff = abs(img1-img2);
imagesc(diff(:,:,30)');
colormap gray
axis equal tight off



%%
%評価プロセス
%メディアン画像読み出し
tmp = load_raw('F:\study_M1\NZ\median\E1_median3.raw','double');
tmp = reshape(tmp,siz);

%リングアーチファクト除去能力評価
subplot(1,3,1);
test = Out(142:162,192:212,36)';
imagesc(test);
colormap gray
caxis([0,0.6])
axis equal tight off
 rectangle('Position',[12,6,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[14,7,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[11,9,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[13,10,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[10,12,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[12,12,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[10,15,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[12,15,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[9,17,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2) 
rectangle('Position',[11,18,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
I = test;
for i=1:size(I,1)-2
for j=1:size(I,2)-2
%Sobel mask for x-direction:
mx=((2*I(i+2,j+1)+I(i+2,j)+I(i+2,j+2))-(2*I(i,j+1)+I(i,j)+I(i,j+2)));
%Sobel mask for y-direction:
my=((2*I(i+1,j+2)+I(i,j+2)+I(i+2,j+2))-(2*I(i+1,j)+I(i,j)+I(i+2,j)));
B(i,j)=sqrt(mx.^2+my.^2);
end
end
ring = B(12,6)+B(14,7)+B(11,9)+B(13,10)+B(10,12)+B(12,12)+B(10,15)+B(12,15)+B(9,17)+B(11,18);
disp(ring)
ring_bi = [B(12,6),B(14,7),B(11,9),B(13,10),B(10,12),B(12,12),B(10,15),B(12,15),B(9,17),B(11,18)];


%エッジ保存評価
subplot(1,3,2);
test2 = Out(230:250,285:305,36)';
imagesc(test2);
 rectangle('Position',[11,3,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[12,5,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[13,7,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[13,8,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[14,11,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[13,10,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[14,15,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[13,19,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
 rectangle('Position',[14,13,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2) 
rectangle('Position',[14,17,1,1]-[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r','LineWidth',2)
colormap gray
caxis([0,0.6])
axis equal tight off
I = test2;
for i=1:size(I,1)-2
for j=1:size(I,2)-2
%Sobel mask for x-direction:
mx=((2*I(i+2,j+1)+I(i+2,j)+I(i+2,j+2))-(2*I(i,j+1)+I(i,j)+I(i,j+2)));
%Sobel mask for y-direction:
my=((2*I(i+1,j+2)+I(i,j+2)+I(i+2,j+2))-(2*I(i+1,j)+I(i,j)+I(i+2,j)));
B(i,j)=sqrt(mx.^2+my.^2);
end
end
edge = B(11,3)+B(12,5)+B(13,7)+B(13,8)+B(14,11)+B(13,10)+B(14,15)+B(13,19)+B(14,13)+B(14,17);
disp(edge)
edge_bi = [B(11,3),B(12,5),B(13,7),B(13,8),B(14,11),B(13,10),B(14,15),B(13,19),B(14,13),B(14,17)];

%ホワイトノイズ除去評価
subplot(1,3,3);
var_bi = Out(260:280,208:228,36)';
imagesc(var_bi);
caxis([0,0.6])
axis equal tight off
%Nで割っている(N-1ではない）
random = std2(var_bi);
disp(random)


%%
%ウィルコクソンの符号順位検定
[p,h,stats] = signrank(ring_bi,ring_pbi);
%F検定
%[h,p] = vartest2(bb,cc);

%%
%箱ひげ図作成
aa = reshape(var_tmp,[441,1]);
value_aa = (aa - mean(aa)).^2;
bb = reshape(var_bi,[441,1]);
value_bb = (bb - mean(bb)).^2;
cc = reshape(var_pbi,[441,1]);
value_cc = (cc - mean(cc)).^2;

boxplot([value_aa,value_bb,value_cc],'Labels',{'Median','Bilateral','Proposed bilateral'});
ylabel('RN')

%boxplot([edge_me,edge_bi,edge_pbi],'Labels',{'Median','Bilateral','Proposed bilateral'});
%ylabel('EP')

%boxplot([ring_me,ring_bi,ring_pbi],'Labels',{'Median','Bilateral','Proposed bilateral'});
%ylabel('RR')

%%
%論文図
imagesc(In(:,:,36)');
colormap gray
caxis([0,0.6])
axis equal tight
rectangle('Position',[142,192,20,20],'FaceColor','none','EdgeColor','g','LineWidth',2)
rectangle('Position',[230,285,20,20],'FaceColor','none','EdgeColor','g','LineWidth',2)
rectangle('Position',[260,208,20,20],'FaceColor','none','EdgeColor','b','LineWidth',2)

%%
%重みの図表示
sig_d = 3;
sig_r = 1.0;
sig_h = 0.003;
w = 9.0;
alpha = 15000;
beta = 100000;

         k =36;j = 202; i = 152; 
        %k =36; j = 290; i =326;
         iMin = max(i-w,1);
         iMax = min(i+w,dim(1));
         jMin = max(j-w,1);
         jMax = min(j+w,dim(2));
         kMin = max(k-w,1);
         kMax = min(k+w,dim(3));
         I = In(iMin:iMax,jMin:jMax,kMin:kMax);
         O = Out(iMin:iMax,jMin:jMax,kMin:kMax);
         Thir_tmp = Thir(iMin:iMax,jMin:jMax,kMin:kMax);
         sig_ddash = powerldash(i,j,k).*alpha + sig_d;
         sig_rdash = powerldash(i,j,k).*beta + sig_r;
         
         Fir = exp(-xyz((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1,(kMin:kMax)-k+w+1)./(2*sig_ddash.^2));
         Sec = exp(-(I-In(i,j,k)).^2./(2*sig_rdash.^2));
         FST = Fir.*Sec.*(Thir_tmp).^2;
         Out(i,j,k) = sum(FST(:).*I(:))/sum(FST(:));
         
         clf
         subplot(2,3,1);
         imagesc(imresize(I(:,:,1+w)',4,'nearest'))
         colormap gray
         caxis([0,0.6])
         axis equal tight off
         rectangle('Position',[9,9,1,1]*4+[0.5,0.5,0,0],'FaceColor','none','EdgeColor','r',...
    'LineWidth',2)

         subplot(2,3,2);
         imagesc(imresize(Fir(:,:,1+w)',4,'nearest'))
         colormap gray
         caxis([0,1])
         axis equal tight off
         
         subplot(2,3,3);
         imagesc(imresize(Sec(:,:,1+w)',4,'nearest'))
         colormap gray
         axis equal tight off
         caxis([0,1])
         
         subplot(2,3,4);
         imagesc(imresize(Thir_tmp(:,:,1+w).^4',4,'nearest'))
         colormap gray
         axis equal tight off
         caxis([0,1])
          
         subplot(2,3,5);
         imagesc(imresize(FST(:,:,1+w)',4,'nearest'))
         colormap gray
         axis equal tight off
         
         subplot(2,3,6);
         imagesc(imresize(O(:,:,1+w)',4,'nearest'))
         colormap gray
         caxis([0,0.6])
         axis equal tight off
         
%%
%濃度プロファイル表示
rangex = [230,330];
rangey = [y-55,y+55];
y =230;
test1 = img_cpu(:,y,65);
test2 = img_cpu(:,y,65);

subplot(3,4,[1,6])
imagesc(img_cpu(:,:,65)');
colormap gray
caxis([0,0.6])
axis equal tight
 rectangle('Position',[1,y-0.5,436,1],'FaceColor','none','EdgeColor','r',...
    'LineWidth',1)
xlim(rangex);
ylim(rangey);

subplot(3,4,[3,8])
imagesc(img_cpu(:,:,12)');
colormap gray
caxis([0,0.6])
axis equal tight
 rectangle('Position',[1,y-0.5,436,1],'FaceColor','none','EdgeColor','r',...
    'LineWidth',1)
xlim(rangex);
ylim(rangey);

M1 = movmean(test1,3);
subplot(3,4,[9,10])
h= plot([test1,M1]);
h(1).Color = 'g';
xlim(rangex);
ylim([-0.1,0.7]);
legend({'Gray value','movemean'});

M2 = movmean(test2,3);
subplot(3,4,[11,12])
h= plot([test2,M2]);
h(1).Color = 'g';
xlim(rangex);
ylim([-0.1,0.7]);
legend({'Gray value','movemean'});

