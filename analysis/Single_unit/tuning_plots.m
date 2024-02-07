clear all
clc

fileinfo = 'E:\forMo\New\Data\Linear_regr\sorted\all_goal\';

files = dir(strcat(fileinfo,'*.mat'));
files_sh = dir(strcat(fileinfo_sh,'*.mat'));

Tri_type = [1 4;2 NaN;3 NaN];

% Event to use
E = 3;
% Times to use
times = 30:100;

% Percentiles
prc = [2.5 50 97.5];
tot_min_sp = [21,44,0;21,63,0;17,73,0;24,81,0;17,59,0;25,62,0;18,19,15;8,16,8;10,12,11;14,12,19;16,14,13;20,5,13;21,12,41];

%%

% alpha = 0.05/length(times);
alpha = 0.05/1;
cumul = 5;

for dv = 1:2
    for se = 1:6
        se_num = (2*se-1):(2*se);

        temp_se = se_num(dv);
        load(strcat(files(temp_se).folder,'\',files(temp_se).name));
            
        ch = size(p_vals,3);

        p_valsm_50 = squeeze(prctile(p_vals(1:4,:,:,:,:),prc(2),4));
        sig_50a = p_valsm_50(:,:,:,:)<alpha;
        sig_50a(5,:,:,:) = (sum(sig_50a(2:4,:,:,:),1)>1);
        indsig_50a = (movmean(sig_50a,cumul,2,'endpoints',0)==1).*sig_50a;
        indsig_50 = movmean(indsig_50a,cumul,2)>0;
 
        indsig_50(:,end+1,:,:) = ones(size(indsig_50(:,1,:,:)));
        
        % Latency of each feature per neuron
        Lat_feat = zeros(ch,4,2);
               
        for ey = 1:2
            for c = 1:ch
                Lat_feat(c,1,ey) = find(squeeze(indsig_50(2,:,c,ey)),1);
                Lat_feat(c,2,ey) = find(squeeze(indsig_50(3,:,c,ey)),1);
                Lat_feat(c,3,ey) = find(squeeze(indsig_50(4,:,c,ey)),1);
                Lat_feat(c,4,ey) = find(squeeze(indsig_50(5,:,c,ey)),1);
            end
        end

        Lat_feat(Lat_feat == size(indsig_50,2)) = NaN;
        indsig_50(:,end,:,:) = [];
        
       cumul_sum(se,dv,:,:) = sum(Lat_feat>0,1);
       
       Tot_lat{se,dv} = Lat_feat;
       Tot_sig_50{se,dv} = indsig_50;
       
        clear Lat_feat
        clear tot_sig
        clear p_vals
        clear p_valsm_50
        clear sig_50a
        clear indsig_50
        
    end
    
end

%% plots

anat = 1;
eypl = 1;
ses2pl = [1:3];

cat_lat = cat(1,Tot_lat{ses2pl,anat});

colm = [0.5 0.5 0.5; 0.00,0.45,0.74; 0.85,0.33,0.10; 0.93,0.69,0.13; 0.49,0.18,0.56];

figure;

for pl = 1:4
    
    temp_lat = cat_lat(~isnan(cat_lat(:,pl,eypl)),pl,eypl);

    subplot(1,4,pl)
    violinplot(temp_lat,[],'ViolinColor', colm(pl+1,:));
    ax1=gca;
    yruler = ax1.YRuler;

    if pl == 1
        ylabel('Time (ms)')
        yticks(0:20:length(times))
        yticklabels(times(1)*20-1000:400:times(end)*20-1000)
        ylim([1 length(times)])
        yline(20)
    else
        ylabel([])
        yticks([])
        yticklabels([])
        yline(20)
        ylim([1 length(times)])
        yruler.Axle.Visible = 'off';
    end

    xticks([])
    xticklabels([])
    title(['N = ', num2str(length(temp_lat))])
    
    clear temp_lat
end

set(gcf,'color','w');
set(gca,'TickLength',[0 0]);
set(gca,'FontSize',8)

set(gcf,'units','points','position',[0,0,300,500])


%% Per session plot with 95% CI
pl_ti = 2:length(times);
anat = 1;
eypl = 1;
ses2pl = [1:3];

colm = [0.5 0.5 0.5; 0.00,0.45,0.74; 0.85,0.33,0.10; 0.93,0.69,0.13; 0.49,0.18,0.56];

cat_lat = cat(1,Tot_lat{ses2pl,anat});
cat_sig50 = cat(3,Tot_sig_50{ses2pl,anat});
neur_num = size(cat_sig50,3);

% cat_sig50 = cat_sig50.*repmat(cat_sig50(3,:,:,eypl),[size(cat_sig50,1),1,1,1]);

sum_catsig = squeeze(sum(cat_sig50(:,:,:,eypl),3));
sum_catsig_prop = sum_catsig/neur_num;

min_lat = min(squeeze(cat_lat(:,:,eypl)),[],2);
[~,srt_idx] = sort(min_lat,'descend');

lab_sig = (squeeze(cat_sig50(2,:,:,eypl))*1)';
lab_sig(squeeze(cat_sig50(3,:,:,eypl)>0)') = 2;
lab_sig(squeeze(cat_sig50(4,:,:,eypl)>0)') = 3;
lab_sig(squeeze(cat_sig50(5,:,:,eypl)>0)') = 4;

lab_srt = lab_sig(srt_idx,:);
idx_sig = sum(lab_srt,2)>0;

figure;

subplot(6,1,1)
plot((sum_catsig_prop(2,pl_ti)>alpha)*2,'.', 'LineStyle', 'none','MarkerSize',8,'color',colm(2,:));
hold on
plot(sum_catsig_prop(3,pl_ti)>alpha,'.', 'LineStyle', 'none','MarkerSize',8,'color',colm(3,:));
plot((sum_catsig_prop(4,pl_ti)>alpha)*3,'.', 'LineStyle', 'none','MarkerSize',8,'color',colm(4,:));
plot((sum_catsig_prop(5,pl_ti)>alpha)*4,'.', 'LineStyle', 'none','MarkerSize',8,'color',colm(5,:));
ylim([0.5 4.5])
xlim([0 length(times)-1])
set(gca,'visible','off')

xticks(0:20:length(times))
xticklabels(times(1)*20-1000:400:times(end)*20-1000)
xline(20)
xline([40,60],'--')

subplot(6,1,[2:3])
plot(sum_catsig_prop(2,pl_ti),'color',colm(2,:),'linewidth',2);
hold on
plot(sum_catsig_prop(3,pl_ti),'color',colm(3,:),'linewidth',2);
plot(sum_catsig_prop(4,pl_ti),'color',colm(4,:),'linewidth',2);
plot(sum_catsig_prop(5,pl_ti),'color',colm(5,:),'linewidth',2);

xticks(0:20:length(times))
xticklabels(times(1)*20-1000:400:times(end)*20-1000)
ylim([0 0.4])
xline(20)
xline([40,60],'--')
yline(alpha)
ylabel('Proportion Tuned')

subplot(6,1,[4:6])
hAxes = gca;
imagesc(hAxes, lab_srt(idx_sig,pl_ti));
% colormap( hAxes ,[0.5 0.5 0.5; 0.00,0.45,0.74; 0.85,0.33,0.10; 0.93,0.69,0.13])
% colormap( hAxes ,[0.5 0.5 0.5; 0.00,0.45,0.74; 0.85,0.33,0.10; 0.49,0.18,0.56; 0.93,0.69,0.13])
colormap( hAxes ,[0.5 0.5 0.5; 0.00,0.45,0.74; 0.85,0.33,0.10; 0.93,0.69,0.13; 0.49,0.18,0.56])
set(gcf,'color','w');
set(gca,'TickLength',[0 0]);
set(gca,'FontSize',8)
xticks(0:20:length(times))
xticklabels(times(1)*20-1000:400:times(end)*20-1000)
xline(20,'color','w')
xline([40,60],'--','color','w')
xlabel('Time (ms) around Goal Onset')
ylabel('Neuron')



set(gcf,'color','w');
% set(gca,'TickLength',[0 0]);
% set(gca,'FontSize',8)
set(gcf,'units','points','position',[0,0,300,600])


xlabel('Time (ms) around Goal Onset')

%%

