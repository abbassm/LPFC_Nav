clear all
clc

% Use chan (1) vs use sorted spikes (0)
usech = 0;

if usech == 0
    fileinfo = '\Data\Neural_data\400ms_20sli\sorted\';
    fileinfo_eye = '\Data\Eye_data_nan\400win_20sli\sorted\';
elseif usech == 1
    fileinfo = '\Data\Neural_data\400ms_20sli\chan\';
    fileinfo_eye = '\Data\Eye_data\400win_20sli\chan\';
end

files = dir(strcat(fileinfo,'*.mat'));
files_eye = dir(strcat(fileinfo_eye,'*.mat'));

Tri_type = [1 4;2 NaN;3 NaN];

% Event to use
E = 3;
arr = 2;
% Times to use
times = 30:100;

% Number of SEs
Num_se = 1;

%%

for i = 1:6

        load(strcat(files_eye(i).folder,'\',files(i).name));
        eye2use = Eye_pos;

        load(strcat(files(i).folder,'\',files(i).name));

        
    %    % Only use if using spikes summed over chan 
        if usech == 1
             Ch1 = unique(Ch1);
             Ch2 = unique(Ch2);
        end


        % Make x and y eye pos last chan of spikerate, 

    %     SpikeRate(:,:,end+1,:) = Eye_pos;

        SpikeRate(:,:,end+1,:) = eye2use(:,:,:,1);
        SpikeRate(:,:,end+1,:) = eye2use(:,:,:,2);

        %% Org data 

        % Remove NaNs
        SpikeRate = SpikeRate(~isnan(Summary_data(:,10)),:,:,:);
        Summary_data = Summary_data(~isnan(Summary_data(:,10)),:,:,:);

        % Only keep "correct" trials
        SpikeRate = SpikeRate(Summary_data(:,6)==1,:,:,:);
        Summary_data = Summary_data(Summary_data(:,6)==1,:);
        
%         % Randomize label
%         Summary_data = Summary_data(randperm(length(Summary_data)),:);

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

        % Label 7 = correct side, label 8 = configuration (1 = LSc/RWc, -1
        % = RSc/LWc), label 9 = chosen col (Sc = -1, Wc = +1)
        Label(:,7) = Label(:,1).*temp_cor;
        Label(:,8) = Label(:,2).*Label(:,7);
        Label(:,9) = Label(:,1).*Label(:,8);

        % Label 10 bin context
        Label(:,10) = Label(:,2)>0;
        
 %      % Add random label to Label 11
        Label(:,11) = Label(randperm(length(Label)),1);
        
        
        % Remove trials with decision times <500 and >1500 ms
        SpikeRate = SpikeRate(Label(:,5)>500 & Label(:,5)<1500,:,:,:);
        Label = Label(Label(:,5)>500 & Label(:,5)<1500,:);

        if arr == 1
          Spikes = SpikeRate(:,times,1:length(Ch1),E);

        elseif arr == 2
          Spikes = SpikeRate(:,times,length(Ch1)+1:length(Ch1)+length(Ch2),E);
        else
          Spikes = SpikeRate(:,times,:,E);

        end

        Spikes(:,:,end+1) = SpikeRate(:,times,end-1,E);
        Spikes(:,:,end+1) = SpikeRate(:,times,end,E);

   
        spikes_tr = Spikes;
        label_tr = Label;

                if arr == 1
                    ch = length(Ch1);
                elseif arr == 2
                    ch = length(Ch2);
                else
                    ch = length(Ch1) + length(Ch2);
                end


                for n = 1:ch

                    temp_unit = zscore(spikes_tr(:,:,n),0,1);


                    for t = 1:length(times)

                        FR = temp_unit(:,t);

                        % Multivariate linear model, predictors = 1 - side 2 - cont 3 -
                        % conf (contxside) 4 - tri type (fixed vs novel)
                        % type 5 - eye x 6 eye y

                         pred = label_tr(:,[1,2,10]);                   
                        pred(:,4) = spikes_tr(:,t,end-1);
                        pred(:,5) = spikes_tr(:,t,end);
% 
                        mdl = fitlm(pred(:,4:5),FR,'interactions');
                        resid = mdl.Residuals.Raw;
                        r2_eye(t,n) = mdl.Rsquared.Ordinary;
                        betas_eye(:,t,n) = mdl.Coefficients.Estimate;
                        p_eye(:,t,n) = mdl.Coefficients.pValue;

                        mdl = fitlm(pred(:,1:3),resid,'interactions');
                        r2(t,n,1) = mdl.Rsquared.Ordinary;
                        betas(:,t,n,1) = mdl.Coefficients.Estimate;
                        p_vals(:,t,n,1) = mdl.Coefficients.pValue;

                        mdl = fitlm(pred(:,1:3),FR,'interactions');
                        r2(t,n,2) = mdl.Rsquared.Ordinary;
                        betas(:,t,n,2) = mdl.Coefficients.Estimate;
                        p_vals(:,t,n,2) = mdl.Coefficients.pValue;

                    end
                end   

            
            if usech ==0
                save(strcat('\Data\Linear_regr\sorted\all_goal\',files(i).name(1),'_sess_',num2str(i),'_arr_',num2str(arr),'_Linear_reg.mat'),'r2_eye','betas_eye','p_eye','r2','betas','p_vals');
            elseif usech ==1
                save(strcat('\Data\Linear_regr\chan\all_goal\',files(i).name(1),'_sess_',num2str(i),'_arr_',num2str(arr),'_Linear_reg.mat'),'r2_eye','betas_eye','p_eye','r2','betas','p_vals');
            end

            clear Label
            clear Spikes
            clear Eye_pos
            clear SpikeRate
            clear Summary_data
            clear spikes_tr
            clear Label_tr
            clear r2_eye
            clear betas_eye
            clear p_eye
            clear r2
            clear betas
            clear p_vals        
end

