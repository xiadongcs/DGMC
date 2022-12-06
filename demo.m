clc
close all

%% 
addpath('data'); addpath('functions'); 
Files = dir(fullfile('data', '*.mat'));
Max_datanum = length(Files);

%% 
for data_num = 1:Max_datanum   
    Dname = Files(data_num).name;
    disp(['***********The test data name is: ***' num2str(data_num) '***'  Dname '****************'])
    load(Dname);
    
    file_path = 'Results/';
    folder_name = Dname(1:end-4);  
    file_path_name = strcat(file_path,folder_name);
    if exist(file_path_name,'dir') == 0   
       mkdir(file_path_name);
    end
    file_mat_path = [file_path_name '/'];
    
    k = 10; lambda = 10.^(-5:0.1:1); r = 1.1:0.1:2;
    time_DGMC = zeros(length(lambda),length(r),length(k));
    Result_DGMC = zeros(7,length(lambda),length(r),length(k));
    for k_i = 1:length(k)
        knn = k(k_i);
        for lambda_i = 1:length(lambda)
            lambda_value = lambda(lambda_i);
            for r_i = 1:length(r)
                r_value = r(r_i);

                tic 
                [lab,~,~] = DGMC(X,Y,1,knn,lambda_value,r_value,0);
                time_DGMC(lambda_i,r_i,k_i) = toc;
                result_DGMC = ClusteringMeasure(Y,lab);
                Result_DGMC(:,lambda_i,r_i,k_i) = result_DGMC'; 
                
                fprintf('knn=%d lambda=%f r=%f\n ',knn,lambda_value,r_value);
                fprintf('DGMC_ACC: %f DGMC_NMI: %f\n',Result_DGMC(1,lambda_i,r_i,k_i),Result_DGMC(2,lambda_i,r_i,k_i));
                
                file_name = Dname;
                save ([file_mat_path,file_name],'Dname','time_DGMC','Result_DGMC');
                
            end
         end
     end  
end