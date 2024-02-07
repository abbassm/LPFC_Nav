clear all
clc

% Use chan (1) vs use sorted spikes (0)
usech = 1;

if usech ==0
    fileinfo = '\Data\Neural_data\400ms_20sli\sorted\';
    fileinfo_eye = '\Data\Eye_data_nan\400win_20sli\sorted\';
elseif usech ==1
    fileinfo = '\Data\Neural_data\400ms_20sli\chan\';
    fileinfo_eye = '\Data\Eye_data_nan\400win_20sli\chan\';
end
    
files = dir(strcat(fileinfo,'*.mat'));
files_eye = dir(strcat(fileinfo_eye,'*.mat'));

Tri_type = [1 4;2 NaN;3 NaN];

% Event to use
E = 3;

% Times to use
times = [30:100];

% Number of SEs
Num_se = 1;

%K-folds
k = 5;

% first 1 - first n_tri, 2 - last n_tri, 3 - all trials
n_tri = 100;
first = 3;

% Trials to use in each group
min2use = 15;

%% Decoding

for i = 1:length(files)
    
    load(strcat(files_eye(i).folder,'\',files(i).name));
    eye2use = Eye_pos;
    
    load(strcat(files(i).folder,'\',files(i).name));
     
    if  files(i).name(1) == 'B'
        tri = 2;
    else
        tri = 3;
    end
    
    if usech == 1
        Ch1 = unique(Ch1);
        Ch2 = unique(Ch2);
    end

    %% Org data 
        
    % Remove NaNs
    SpikeRate = SpikeRate(~isnan(Summary_data(:,10)),:,:,:);
    Summary_data = Summary_data(~isnan(Summary_data(:,10)),:,:,:);
    
    % Only keep "correct" trials
    SpikeRate = SpikeRate(Summary_data(:,6)==1,:,:,:);
    Summary_data = Summary_data(Summary_data(:,6)==1,:);
    
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

    % Remove trials with decision times <500 and >1500 ms
    SpikeRate = SpikeRate(Label(:,5)>500 & Label(:,5)<1500,:,:,:);
    Label = Label(Label(:,5)>500 & Label(:,5)<1500,:);

    Spikes = SpikeRate(:,times,:,E);

 %%  
    for tr = 1:tri        
        spikes_tr = Spikes(or(Label(:,4)==Tri_type(tr,1),Label(:,4)==Tri_type(tr,2)),:,:);
        Label_tr = Label(or(Label(:,4)==Tri_type(tr,1),Label(:,4)==Tri_type(tr,2)),:);
        
         if first == 1
            spikes_tr = spikes_tr(1:n_tri,:,:);
            Label_tr = Label_tr(1:n_tri,:);
         elseif first == 2
            spikes_tr = spikes_tr(end-n_tri+1:end,:,:);
            Label_tr = Label_tr(end-n_tri+1:end,:);
         end
         
         for r = 1:100
        
            for t = 1:length(times)
         
                 Label_temp = Label_tr((sum(isnan(spikes_tr(:,t,:)),3))==0,:);
                 spikes_t = squeeze(spikes_tr((sum(isnan(spikes_tr(:,t,:)),3))==0,t,:));
                 
                 Label_temp = Label_temp(~isnan(sum(spikes_t,2)),:);
                 spikes_t = spikes_t(~isnan(sum(spikes_t,2)),:);

                % Min number of trials
                 hist_tr = histc(Label_temp(:,10),[1;2;3;4]);
        %         min_tri = min(hist_tr);

   
                % Min_tri at most min2use
                if min(hist_tr)<min2use
                     min_tri = min(hist_tr);
                else
                     min_tri = min2use;
                end
    
                % Keep track of min trial number
                tot_min_tr(t,tr,i,r) = min(hist_tr);
            
            % Equal number of groups
            
            for g = 1:4
                
                temp_idx = randperm(hist_tr(g),min_tri);
                
                temp_lab =  Label_temp(Label_temp(:,10)==g,:);
                temp_sp = spikes_t(Label_temp(:,10)==g,:);
                
                lab_eq(:,:,g) = temp_lab(temp_idx,:);
                lab_sp(:,:,g) = temp_sp(temp_idx,:);
                
                clear temp_idx
                clear temp_lab
                clear temp_sp
            end
                 
            kfold_ind = crossvalind('Kfold',min_tri,k);

                for k_it = 1:k
                    
                    
                     tr_temp_sp = squeeze([lab_sp(kfold_ind~=k_it,:,1);lab_sp(kfold_ind~=k_it,:,2);lab_sp(kfold_ind~=k_it,:,3);lab_sp(kfold_ind~=k_it,:,4)]);
                     tr_temp_lab = [lab_eq(kfold_ind~=k_it,:,1);lab_eq(kfold_ind~=k_it,:,2);lab_eq(kfold_ind~=k_it,:,3);lab_eq(kfold_ind~=k_it,:,4)];
                        
                     te_temp_sp = squeeze([lab_sp(kfold_ind==k_it,:,1);lab_sp(kfold_ind==k_it,:,2);lab_sp(kfold_ind==k_it,:,3);lab_sp(kfold_ind==k_it,:,4)]);
                     te_temp_lab = [lab_eq(kfold_ind==k_it,:,1);lab_eq(kfold_ind==k_it,:,2);lab_eq(kfold_ind==k_it,:,3);lab_eq(kfold_ind==k_it,:,4)];

                     % z-scoring
                    tr_temp_sp = zscore(tr_temp_sp,0,1);
                    te_temp_sp = zscore(te_temp_sp,0,1);

                    % Spikes

                    for ar = 1:3
                        
                        if ar ==1                             
                            temp_tr = tr_temp_sp(:,1:length(Ch1));
                            temp_te = te_temp_sp(:,1:length(Ch1));
                        elseif ar ==2 
                            temp_tr = tr_temp_sp(:,length(Ch1)+1:length(Ch1)+length(Ch2));
                            temp_te = te_temp_sp(:,length(Ch1)+1:length(Ch1)+length(Ch2));
                        elseif ar ==3
                            temp_tr = tr_temp_sp(:,length(Ch1)+length(Ch2)+1:end);
                            temp_te = te_temp_sp(:,length(Ch1)+length(Ch2)+1:end);
                        end
                        
                        mdl = fitcsvm(temp_tr,tr_temp_lab(:,1));
                        temp_l =  predict(mdl,temp_te);
                        acc_side(t,k_it,r,tr,i,ar) = sum(temp_l==te_temp_lab(:,1))/length(te_temp_lab);

                        mdl = fitcsvm(temp_tr,tr_temp_lab(:,2));
                        temp_l =  predict(mdl,temp_te);
                        acc_cont(t,k_it,r,tr,i,ar) = sum(temp_l==te_temp_lab(:,2))/length(te_temp_lab);

                        mdl = fitcsvm(temp_tr,tr_temp_lab(:,8));
                        temp_l =  predict(mdl,temp_te);
                        acc_conf(t,k_it,r,tr,i,ar) = sum(temp_l==te_temp_lab(:,8))/length(te_temp_lab);

                        mdl = fitcsvm(temp_tr,tr_temp_lab(randperm(length(tr_temp_lab(:,2))),1));
                        temp_l =  predict(mdl,temp_te);
                        acc_shuff(t,k_it,r,tr,i,ar) = sum(temp_l==(te_temp_lab(randperm(length(te_temp_lab(:,2))),1)))/length(te_temp_lab);
                        clear temp_tr
                        clear temp_te
                        
                    end
                    
                end
                
                 clear lab_eq
                 clear lab_sp
                 clear Label_temp
                 clear spikes_t
    %                 
             end
        end
        

        
    end
    clear Label
    clear Spikes
    clear Eye_pos
       
end
