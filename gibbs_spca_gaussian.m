function [fa, samples] = gibbs_spca_gaussian(y, pr, op, initial_fa)
% -- Likelihood:
%    p(y_ij|x_i,w_j,sigma2_j) = N(y_ij|w_j'x_i, sigma2_j)
% -- Prior:
% p(x_i) = N(x_i|0, I)
% p(w_jk|gamma_j=1) = Normal(w_jk|0, tau2)
% p(w_jk|gamma_j=0) = delta(w_jk)
% p(gamma_jk|rho) = Bernoulli(gamma_jk|rho)
%
% Tomi Peltola, tomi.peltola@aalto.fi
% http://becs.aalto.fi/en/research/bayes/epwx/

[n, m] = size(y);
K = pr.K;

x = initial_fa.x.Mean; % K x n
w = initial_fa.w.Mean; % K x m
gam = initial_fa.P_gamma == 1; % K x m

log_rho_tau2_terms = log(pr.rho) - log1p(-pr.rho) - 0.5 * log(pr.tau2);
sigma = sqrt(pr.sigma2);

samples.x = zeros(K, n, op.nsamples);
samples.w = zeros(K, m, op.nsamples);
samples.gam = false(K, m, op.nsamples);
n_upd = 0;

%% sampling
for iter = 1:op.nsamples
    %% sample gamma and w
    for j = 1:m
        yy = y(:, j);
        [logz_old, xx_chol_old, v_old] = compute_det_and_exp_terms(x, yy, gam(:, j), pr);
        
        % sample gamma
        for k = randperm(K)
            gam_new = gam(:, j);
            if gam(k, j)
                % already in model
                gam_new(k) = false;
                coeff = 1;
            else
                % not in model
                gam_new(k) = true;
                coeff = -1;
            end
            
            [logz_new, xx_chol_new, v_new] = compute_det_and_exp_terms(x, yy, gam_new, pr);
            % AB is the log Bayes factor for the larger model
            % (gam(k, j) = 1) compared to the smaller (gam(k, j) = 0)
            AB = log_rho_tau2_terms + coeff * (logz_old - logz_new);

            % collapsed conditional probability of gam(k, j) = 1
            prob = 1 / (1 + exp(-AB));
            
            gam(k, j) = rand() < prob;
            
            % did the gam change? if yes, update current logz
            if gam(k, j) == gam_new(k)
                logz_old = logz_new;
                xx_chol_old = xx_chol_new;
                v_old = v_new;
                
                n_upd = n_upd + 1;
            end
        end
        
        % sample w
        w(~gam(:, j), j) = 0;
        if sum(gam(:, j)) > 0
            w(gam(:, j), j) = xx_chol_old' \ (sigma * randn(sum(gam(:, j)), 1) + v_old);
        end
    end
    
    %% sample x
    ww = w * w';
    ww(1:(size(ww, 1)+1):end) = ww(1:(size(ww, 1)+1):end) + pr.sigma2;
    
    ww_chol = chol(ww, 'lower');
    for i = 1:n
        x(:, i) = ww_chol' \ (sigma * randn(K, 1) + ww_chol \ (w * y(i, :)'));
    end
    
    %% collect samples
    samples.x(:, :, iter) = x;
    samples.w(:, :, iter) = w;
    samples.gam(:, :, iter) = gam;
    
    %%
    if op.verbosity > 0 && mod(iter, op.verbosity) == 0
        fprintf(1, '%d: upd.rate: %.3f\n', iter, n_upd / (iter * m * K));
    end
end

%% compute posterior averages
fa.P_gamma = mean(samples.gam(:, :, (op.nwarmup+1):end), 3);
fa.w.Mean = mean(samples.w(:, :, (op.nwarmup+1):end), 3);
fa.x.Mean = mean(samples.x(:, :, (op.nwarmup+1):end), 3);
fa.w.Var = var(samples.w(:, :, (op.nwarmup+1):end), 0, 3);
fa.x.Var = var(samples.x(:, :, (op.nwarmup+1):end), 0, 3);

end


function [logz, xx_chol, v] = compute_det_and_exp_terms(x, yy, gam, pr)

if sum(gam) == 0
    logz = 0;
    xx_chol = [];
    v = [];
    return
end

x_gam = x(gam, :);
xx = x_gam * x_gam';
xx(1:(size(xx, 1)+1):end) = xx(1:(size(xx, 1)+1):end) + pr.sigma2 / pr.tau2;
xx_chol = chol(xx, 'lower');

v = xx_chol \ (x_gam * yy);
logz = 0.5 * (v' * v) / pr.sigma2 - sum(log(diag(xx_chol)));

end
