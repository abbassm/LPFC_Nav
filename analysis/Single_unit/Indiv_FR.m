clear all
clc

load('\Data\Neural_data\Rasters\B_20171111_Summary.mat')

Num_units = size(Spikes,3);
Tri_type = [1 4;2 NaN;3 NaN];

%% Event and channel to plot

E = 3;
Ch = [32:33];

% Eye regression - 1 = yes
Eye_R = 0;

% Trial dividing early vs late, tri2plt (1 - early, 2 - late, 3 - all)
tri2plt = 3;
tridiv = 250;

tr = 2;
t1 = 800;
t2 = 2200;

Num_se = 1;
gwin = 100;

% Org data

n = t2-t1;

Spike2use = squeeze(sum(Spikes(:,:,Ch,E),3)>0);
Eye2use = squeeze(Eye_pos(:,:,E,:));

clear Spike_gaus
clear Eye_gaus

% Reminder - Consider what to do with NaN in eye data - change to 0 for now
Eye2use(isnan(Eye2use)) = 0;

for i = 1:length(Summary_data)
    Spike_gaus(i,:,1) = conv(Spike2use(i,:),gausswin(gwin),'same')./(gwin/1000);
    Eye_gaus(i,:,1,1) = conv(Eye2use(i,:,1),gausswin(gwin),'same')./(gwin/1000);
    Eye_gaus(i,:,2,1) = conv(Eye2use(i,:,2),gausswin(gwin),'same')./(gwin/1000);
    
end

Spike_gaus(:,:,2) = Spike2use;
Eye_gaus(:,:,:,2) = Eye2use;

Spike_gaus = Spike_gaus(:,t1+1:t2,:);
Eye_gaus = Eye_gaus(:,t1+1:t2,:,:);

clear SpikeRate
clear Summary_data2
clear Label
clear spike_m 
clear spike_se
clear inBetween
clear eye_m
clear eye_se
clear inBeteye
clear x2

% Remove NaNs
SpikeRate = Spike_gaus(~isnan(Summary_data(:,10)),:,:);
Eye_dat = Eye_gaus(~isnan(Summary_data(:,10)),:,:,:);
Summary_data2 = Summary_data(~isnan(Summary_data(:,10)),:,:,:);


% Labels = side, cont, cor/err, tri type, time from goal onset to
% decision, idx
Label(:,1) = Summary_data2(:,4);
Label(:,2) = Summary_data2(:,3);
Label(:,3) = Summary_data2(:,6);
Label(:,4) = Summary_data2(:,2);
Label(:,5) = Summary_data2(:,10)-Summary_data2(:,9);
Label(:,6) = 1:length(Label);

Label(Label(:,1)==0,1) = repmat(-1,sum(Label(:,1)==0),1);
Label(Label(:,2)==0,2) = repmat(-1,sum(Label(:,2)==0),1);

temp_cor = Label(:,3);
temp_cor(temp_cor==0) = repmat(-1,sum(temp_cor==0),1);

% Label 7 = correct side, label 8 = configuration (1 = LSc/RWc, -1 = RSc/LWc)
Label(:,7) = Label(:,1).*temp_cor;
Label(:,8) = Label(:,2).*Label(:,7);
Label(:,9) = Label(:,1).*Label(:,8);
  

% Label 10 - 1 St Sc L, 2 St Sc R, 3 St Wc L, 4 St Wc R, 5 Wo Sc L,
    % 6 Wo Sc R, 7 Wo Wc L, 8 Wo Wc R

% Key in order of Context, Colour and Side for all 8 groups
key_grps = [-1 -1 -1; -1 -1 1; -1 1 -1; -1 1 1; 1 -1 -1; 1 -1 1; 1 1 -1; 1 1 1];

for kgr = 1:length(key_grps)

    temp_lab10 = (Label(:,2) == key_grps(kgr,1) & Label(:,9) == key_grps(kgr,2) & Label(:,1) == key_grps(kgr,3))*kgr;

    if kgr == 1
        Label(:,10) =  temp_lab10;
    else
        Label(:,10) =  Label(:,10) + temp_lab10;
    end

    clear temp_lab10;
end

% Remove trials with decision times <500 and >1500 ms
SpikeRate = SpikeRate(Label(:,5)>500 & Label(:,5)<1500,:,:);
Eye_dat = Eye_dat(Label(:,5)>500 & Label(:,5)<1500,:,:,:);
Label = Label(Label(:,5)>500 & Label(:,5)<1500,:);

SpikeRate =SpikeRate(or(Label(:,4)==Tri_type(tr,1),Label(:,4)==Tri_type(tr,2)),:,:);
Eye_dat =Eye_dat(or(Label(:,4)==Tri_type(tr,1),Label(:,4)==Tri_type(tr,2)),:,:,:);
Label = Label(or(Label(:,4)==Tri_type(tr,1),Label(:,4)==Tri_type(tr,2)),:);

if Eye_R == 1
    
    for t = 1:n

        temp_FR = squeeze(SpikeRate(:,t,1));

        temp_eye = squeeze(Eye_dat(:,t,:,1));

        mdl = fitlm(temp_eye,temp_FR,'interactions');
        SpikeRate(:,t,1) = mdl.Residuals.Raw;

        clear temp_eye
        clear temp_FR

    end

end

Lab_ear = Label(1:tridiv,:);
Spike_ear = SpikeRate(1:tridiv,:,:);
Eye_ear = Eye_dat(1:tridiv,:,:,:);

Lab_lat = Label(tridiv:end,:);
Spike_lat = SpikeRate(tridiv:end,:,:);
Eye_lat = Eye_dat(tridiv:end,:,:,:);

% Mean + std of cat in lab 10. 3rd dim - 1 = early, 2 = late, 3 = all

for mc = 1:length(key_grps)

    spike_m(mc,:,1) = mean(Spike_ear(Lab_ear(:,10)==mc,:,1),1);
    spike_se(mc,:,1) = Num_se*std(Spike_ear(Lab_ear(:,10)==mc,:,1),0,1)./sqrt(sum(Lab_ear(:,10)==mc));
    inBetween(mc,:,1) =  [(spike_m(mc,:,1)-spike_se(mc,:,1)), (fliplr((spike_m(mc,:,1)+spike_se(mc,:,1))))]; 

    spike_m(mc,:,2) = mean(Spike_lat(Lab_lat(:,10)==mc,:,1),1);
    spike_se(mc,:,2) = Num_se*std(Spike_lat(Lab_lat(:,10)==mc,:,1),0,1)./sqrt(sum(Lab_lat(:,10)==mc));
    inBetween(mc,:,2) =  [(spike_m(mc,:,2)-spike_se(mc,:,2)), (fliplr((spike_m(mc,:,2)+spike_se(mc,:,2))))]; 

    spike_m(mc,:,3) = mean(SpikeRate(Label(:,10)==mc,:,1),1);
    spike_se(mc,:,3) = Num_se*std(SpikeRate(Label(:,10)==mc,:,1),0,1)./sqrt(sum(Label(:,10)==mc));
    inBetween(mc,:,3) =  [(spike_m(mc,:,3)-spike_se(mc,:,3)), (fliplr((spike_m(mc,:,3)+spike_se(mc,:,3))))]; 

end

% Finding mean of x value
for mc = 1:length(key_grps)

    eye_m(mc,:,1) = mean(Eye_ear(Lab_ear(:,10)==mc,:,1,1),1);
    eye_se(mc,:,1) = Num_se*std(Eye_ear(Lab_ear(:,10)==mc,:,1,1),0,1)./sqrt(sum(Lab_ear(:,10)==mc));
    inBeteye(mc,:,1) =  [(eye_m(mc,:,1)-eye_se(mc,:,1)), (fliplr((eye_m(mc,:,1)+eye_se(mc,:,1))))]; 

    eye_m(mc,:,2) = mean(Eye_lat(Lab_lat(:,10)==mc,:,1,1),1);
    eye_se(mc,:,2) = Num_se*std(Eye_lat(Lab_lat(:,10)==mc,:,1,1),0,1)./sqrt(sum(Lab_lat(:,10)==mc));
    inBeteye(mc,:,2) =  [(eye_m(mc,:,2)-eye_se(mc,:,2)), (fliplr((eye_m(mc,:,2)+eye_se(mc,:,2))))]; 

    eye_m(mc,:,3) = mean(Eye_dat(Label(:,10)==mc,:,1,1),1);
    eye_se(mc,:,3) = Num_se*std(Eye_dat(Label(:,10)==mc,:,1,1),0,1)./sqrt(sum(Label(:,10)==mc));
    inBeteye(mc,:,3) =  [(eye_m(mc,:,3)-eye_se(mc,:,3)), (fliplr((eye_m(mc,:,3)+eye_se(mc,:,3))))]; 

end


x2 = [1:n, fliplr(1:n)];


%

ord2use = [1,4;2,3;8,5;7,6];
titles = {'Steel Order 1','Steel Order 2','Wood Order 1 ','Wood Order 2'};


figure; 

for fig = 1:4
    
    subplot(4,2,fig*2)
    
    
    plot(1:n,spike_m(ord2use(fig,1),:,tri2plt),'color','g');
    hold on
    plot(1:n,spike_m(ord2use(fig,2),:,tri2plt),'color','r');
    fill(x2, inBetween(ord2use(fig,1),:,tri2plt), 'g','FaceAlpha',0.2,'EdgeColor','none');
    fill(x2, inBetween(ord2use(fig,2),:,tri2plt), 'r','FaceAlpha',0.2,'EdgeColor','none');
    
   xticks(0:200:n)
    xticklabels(t1-1200:200:t2 - 1200)
    xline(1200-t1)
    title(titles{fig})
    ylim([0 max(max(max(inBetween)))]);
end



for fig = 1:4
    
    subplot(4,2,fig*2-1)
    
    
    plot(1:n,eye_m(ord2use(fig,1),:,tri2plt),'color','g');
    hold on
    plot(1:n,eye_m(ord2use(fig,2),:,tri2plt),'color','r');
    fill(x2, inBeteye(ord2use(fig,1),:,tri2plt), 'g','FaceAlpha',0.2,'EdgeColor','none');
    fill(x2, inBeteye(ord2use(fig,2),:,tri2plt), 'r','FaceAlpha',0.2,'EdgeColor','none');
    
   xticks(0:200:n)
    xticklabels(t1-1200:200:t2 - 1200)
    xline(1200-t1)
    title(titles{fig})
    ylim([min(min(min(inBeteye))) max(max(max(inBeteye)))]);
end


set(gcf,'color','w');
set(gca,'TickLength',[0 0]);
set(gca,'FontSize',8)

