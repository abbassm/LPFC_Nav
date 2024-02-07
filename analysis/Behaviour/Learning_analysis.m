clear all
clc

fileinfo = '\Data\Eye_data\400win_20sli\chan\';
files = dir(strcat(fileinfo,'*.mat'));

Tri_type = [1 4;2 NaN;3 NaN];

% Event to use
E = 3;

% Times to use
times = 1:140;

%%
for i = 1:length(files)
    load(strcat(files(i).folder,'\',files(i).name));
    if  files(i).name(1) == 'B'
        tri = 2;
    else
        tri = 3;
    end
    
    %% Org data 
    
    % Remove NaNs
    Eye_pos = Eye_pos(~isnan(Summary_data(:,10)),:,:,:);
    Summary_data = Summary_data(~isnan(Summary_data(:,10)),:,:,:);

    % Labels = side, cont, cor/err, tri type, time from goal onset to
    % decision, idx
    Label(:,1) = Summary_data(:,4);
    Label(:,2) = Summary_data(:,3);
    Label(:,3) = Summary_data(:,6);
    Label(:,4) = Summary_data(:,2);
    Label(:,5) = Summary_data(:,10)-Summary_data(:,9);
    Label(:,6) = 1:length(Label);
    
    Label(Label(:,1)==0,1) = repmat(-1,sum(Label(:,1)==0),1);
    Label(Label(:,2)==0,2) = repmat(-1,sum(Label(:,2)==0),1);

    temp_cor = Label(:,3);
    temp_cor(temp_cor==0) = repmat(-1,sum(temp_cor==0),1);

    % Label 7 = correct side, label 8 = configuration (1 = LSc/RWc, -1 = RSc/LWc)
    Label(:,7) = Label(:,1).*temp_cor;
    Label(:,8) = Label(:,2).*Label(:,7);
    Label(:,9) = Label(:,1).*Label(:,8);
    
    % Label 10 - 1 - St conf 1, 2 - St conf 2, 3 - W conf 1, 4 - W conf 2
    Label(:,10) = (Label(:,2) == -1 & Label(:,8) == 1)*1 + (Label(:,2) == -1 & Label(:,8) == -1)*2 + (Label(:,2) == 1 & Label(:,8) == 1)*3 + (Label(:,2) == 1 & Label(:,8) == -1)*4;
    
%    % Add random label to Label 11
     Label(:,11) = Label(randperm(size(Label,1)),1);
     
     % Time diff  
    Label(:,12) = Summary_data(:,8)-Summary_data(:,7);
    Label(:,13) = Summary_data(:,9)-Summary_data(:,8);
    Label(:,14) = Summary_data(:,11)-Summary_data(:,10);
                    
    % Remove trials with decision times <500 and >1500 ms
    Eye_pos = squeeze(Eye_pos(Label(:,5)>500 & Label(:,5)<1500,:,E,:));
    Label = Label(Label(:,5)>500 & Label(:,5)<1500,:);
         
     for tr = 1:tri
         
         Label_tr = Label(or(Label(:,4)==Tri_type(tr,1),Label(:,4)==Tri_type(tr,2)),:);
         Eye_tr = Eye_pos(or(Label(:,4)==Tri_type(tr,1),Label(:,4)==Tri_type(tr,2)),:,:);
         
         lab_cor = Label_tr(Label_tr(:,3)==1,:);
         lab_incor = Label_tr(Label_tr(:,3)==0,:);
         
         tot_tri_num(i,tr) = length(lab_cor);
         tot_inc_tri_num(i,tr) = length(lab_incor);
         min_tri_cond(i,tr) = min(histc(lab_cor(:,10),[1;2;3;4]));
         
         [prob_all{i,tr}(1,:),prob_all{i,tr}(2,:),prob_all{i,tr}(3,:)] = runanalysis(Label_tr(:,3),1,0.5);
         
         time_dif{i,tr} =  [Label_tr(:,12:13),Label_tr(:,5),Label_tr(:,14)];
         
         clear lab_cor
        
     end
    clear Label
    clear Label_tr
    clear Summary_data
    clear temp_cor
    
end

%%
% sessions and blocks to plot
tr = 3:3;

figure
count = 1;
for sess = [7:13]
    
    
subplot(4,2,count)
prob2use = prob_all(sess,tr);

min_tr = min(min(cell2mat(cellfun(@length,prob2use,'uni',false))));

for bl = tr(1):tr(end)
    for i = sess
         prob(:,1:min_tr,i,bl) = prob_all{i,bl}(:,1:min_tr);
    end
end

prob = squeeze(mean(prob(:,:,sess,tr),4));

prob_m = squeeze(mean(prob,3));

inBetween = [prob_m(3,:), fliplr(prob_m(1,:))];
x2 = [1:length(prob_m), fliplr(1:length(prob_m))];

% figure;
plot(1:length(prob_m),prob_m(2,:),'Color',[0,0,0]);
hold on
fill(x2, inBetween, [0.2,0.2,0.2],'FaceAlpha',0.2,'EdgeColor','none');
ylim([0.2 1])
xlabel('Trial Number')
ylabel('Proportion Correct')
% title('Novel 2 Blocks')
yline(0.5)

title(['Correct trials = ',num2str(tot_tri_num(sess,tr)),' Min/cond = ', num2str(min_tri_cond(sess,tr))])

set(gcf,'color','w');
set(gca,'TickLength',[0 0]);

% set(gcf,'units','points','position',[0,0,400,300])

clear prob
clear prob_m

count = count+1;
end


set(gcf,'Position',[0 0 300 930])



%%


tot_propcor = tot_tri_num./(tot_tri_num+tot_inc_tri_num);

mean_prop(:,1) = mean(tot_propcor(1:6,1:2),[1 2]);
mean_prop(:,2) = mean(tot_propcor(7:13,1:3),[1 2]);

se_prop(:,1) =  std(tot_propcor(1:6,1:2),[],[1 2])./sqrt(12);
se_prop(:,2) =  std(tot_propcor(7:13,1:3),[],[1 2])./sqrt(18);

