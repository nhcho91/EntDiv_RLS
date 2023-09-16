% Need to run this file with its enclosing folder as the working directory
fileInfo = dir(matlab.desktop.editor.getActiveFilename);
cd(fileInfo.folder);

clc
clear
close all
init_path

%% Load Data
load('CheetahSysID.mat');
model = Cheetah3LegModel();

%% Joint Space Inertia Matrix Positive Definiteness Check -- w.r.t. running estimate
load("theta_hat.mat") % <---- Obtained from Julia code

test_summary = zeros(2, size(theta_hat_final_RLS_l2, 2));
for i_alg = 1 : 2
    for i_f = 1 : size(theta_hat_final_RLS_l2, 2)
        if i_alg == 1
            theta = theta_hat_RLS_l2{i_f};
        elseif i_alg == 2
            theta = theta_hat_RLS_ldetdiv{i_f};
        end

        dt_resample = 1 / 10^(i_f-1);
        dt_save = 1e-3;
        t = 0:dt_save:28;
        id_t_valid = find(t>=0.05 & t<28-0.05); 
        id_t_cons = id_t_valid(1:dt_resample/dt_save:end);
        q_cons = q(id_t_cons);
        qd_cons = qd(id_t_cons);

        M = cell(length(q_cons),1);
        TF_M_pd = zeros(length(q_cons),1);
        for i = 1 : length(q_cons)
            Phi = reshape(theta{i}(1:end-6), 10, []);
            for j = 1 : model.NB
                model.I{j} = inertiaVecToMat(Phi(:,j));
                model.I_rotor{j} = inertiaVecToMat(Phi(:,j+3));
            end

            [H,C,info] = HandC( model, q_cons{i}, qd_cons{i} );
            M{i} = H;

            TF_M_pd(i) = issymmetric(H) && isreal(eig(H)) && prod(eig(H) > 0);
        end
        figure; plot(TF_M_pd), title(sprintf("i_alg = %d, i_f = %d", i_alg, i_f))
        test_summary(i_alg, i_f) = prod(TF_M_pd);
    end
end

%% Joint Space Inertia Matrix Positive Definiteness Check -- w.r.t. final estimate
load("theta_hat.mat") % <---- Obtained from Julia code

test_summary = zeros(2, size(theta_hat_final_RLS_l2, 2));
for i_alg = 1 : 2
    if i_alg == 1
        Phi = theta_hat_final_RLS_l2(1:60,:);
    elseif i_alg == 2
        Phi = theta_hat_final_RLS_ldetdiv(1:60,:);
    end

    for i_f = 1 : size(theta_hat_final_RLS_l2, 2)
        Phi_f = reshape(Phi(:, i_f), 10, []);
        for i = 1 : model.NB
            model.I{i} = inertiaVecToMat(Phi_f(:,i));
            model.I_rotor{i} = inertiaVecToMat(Phi_f(:,i+3));
        end

        M = cell(length(q),1);
        TF_M_pd = zeros(length(q),1);
        for i = 1 : length(q)
            [H,C,info] = HandC( model, q{i}, qd{i} );
            M{i} = H;

            TF_M_pd(i) = issymmetric(H) & isreal(eig(H)) & prod(eig(H) > 0);
        end
        figure; plot(TF_M_pd), title(sprintf("i_alg = %d, i_f = %d", i_alg, i_f))
        test_summary(i_alg, i_f) = prod(TF_M_pd);
    end
end