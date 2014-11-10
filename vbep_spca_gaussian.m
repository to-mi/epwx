function [fa, si, converged] = vbep_spca_gaussian(y, pr, op)
% -- Likelihood:
%    p(y_ij|x_i,w_j,sigma2_j) = N(y_ij|w_j'x_i, sigma2_j)
% -- Prior:
% p(x_i) = N(x_i|0, I)
% p(w_jk|gamma_j=1) = Normal(w_jk|0, tau2)
% p(w_jk|gamma_j=0) = delta(w_jk)
% p(gamma_jk|rho) = Bernoulli(gamma_jk|rho)
% -- Approximation;
% q(w_j) = Normal(w_j|Mean_w_j, Var_w_j), Var_w_j = Tau_w_j^-1
% q(x_i) = Normal(x_i|Mean_x_i, Var_x_i), Var_x_i = Tau_x_i^-1
% q(gamma_j) = \prod Bernoulli(\gamma_jk|p_gamma_jk)
%
% Tomi Peltola, tomi.peltola@aalto.fi
% http://becs.aalto.fi/en/research/bayes/epwx/

[n, m] = size(y);

K = pr.K;
pr.n = n;
pr.m = m;

%% initialize (using given initialization or using pca)
if nargin < 4
    [coeff, score] = pca(y, 'Centered', false, 'NumComponents', K);
    s = 1 / std(score(:));
    coeff = coeff / s;
    score = score * s;
    
    fa_or_si_init.type = 'fa';
    fa_or_si_init.w.Mean = coeff';
    fa_or_si_init.w.Tau = repmat(1/pr.tau2 * eye(K), [1 1 m]);
    fa_or_si_init.x.Mean = score';
    fa_or_si_init.x.Tau = repmat(eye(K), [1 1 n]);
end

switch fa_or_si_init.type
    case 'si'
        si.lik.x.Tau = fa_or_si_init.lik.x.Tau;
        si.lik.x.Mu = fa_or_si_init.lik.x.Mu;
        si.lik.w.Tau = fa_or_si_init.lik.w.Tau;
        si.lik.w.Mu = fa_or_si_init.lik.w.Mu;
    case 'fa'
        % compute Mus from Means
        fa_or_si_init.x.Mu = zeros(size(fa_or_si_init.x.Mean));
        fa_or_si_init.w.Mu = zeros(size(fa_or_si_init.w.Mean));
        for i = 1:n
            fa_or_si_init.x.Mu(:, i) = fa_or_si_init.x.Tau(:, :, i) * fa_or_si_init.x.Mean(:, i);
        end
        for i = 1:m
            fa_or_si_init.w.Mu(:, i) = fa_or_si_init.w.Tau(:, :, i) * fa_or_si_init.w.Mean(:, i);
        end
        
        % compute sites from full approximation (assume same form of
        % approximation as used here)
        si.lik.x.Tau = bsxfun(@minus, fa_or_si_init.x.Tau, eye(K));
        si.lik.x.Mu = fa_or_si_init.x.Mu;
        si.lik.w.Tau = bsxfun(@minus, fa_or_si_init.w.Tau, 1/pr.tau2 * eye(K));
        si.lik.w.Mu = fa_or_si_init.w.Mu;
    otherwise
        error('unknown initialization');
end

si.prior.w.mu = zeros(K, m);
si.prior.w.tau = 1/pr.tau2 * ones(K, m);
si.prior.gamma.a = ones(K, m);
si.prior.gamma.b = ones(K, m);

% full approximation
fa = compute_full_approximation(si, pr);

% convergence diagnostics
conv.z_old = Inf * ones(K, m);
conv.P_gamma_old = Inf * ones(size(fa.P_gamma));

%% loop parallel EP
for iter = 1:op.max_iter
    %% likelihood updates using VB
    % update sites of w
    for j = 1:m
        si.lik.w.Tau(:, :, j) = (sum(fa.x.Cov, 3) + fa.x.Mean * fa.x.Mean') / pr.sigma2;
        si.lik.w.Mu(:, j) = fa.x.Mean * y(:, j) / pr.sigma2;
    end
    
    fa = compute_full_approximation(si, pr);
    
    % update sites of x
    for i = 1:n
        si.lik.x.Tau(:, :, i) = (sum(fa.w.Cov, 3) + fa.w.Mean * fa.w.Mean') / pr.sigma2;
        si.lik.x.Mu(:, i) = fa.w.Mean * y(i, :)' / pr.sigma2;
    end
    
    %% full approx update
    fa = compute_full_approximation(si, pr);
    
    %% prior updates
    % cavity
    ca_prior = compute_prior_cavity(fa, si.prior, pr);
    
    % moments of tilted dists
    [ti_prior, z] = compute_prior_tilt(ca_prior, pr);
    
    % site updates
    si.prior = site_updates_prior(si.prior, ca_prior, ti_prior, op);

    %% full approx update
    fa = compute_full_approximation(si, pr);
    
    %% show progress and check for convergence
    [converged, conv] = report_progress_and_check_convergence(conv, iter, z, fa, op);
    if converged
        break
    end
    
    %% update damp
    op.damp = op.damp * op.damp_decay;
end

end



function ca = compute_prior_cavity(fa, si, pr)

% loop through output variables
ca.w.mean = zeros(pr.K, pr.m);
ca.w.tau = zeros(pr.K, pr.m);

for i = 1:pr.m
    var_w_j = diag(fa.w.Cov(:, :, i));

    denom = (1 - si.w.tau(:, i) .* var_w_j);
    ca.w.tau(:, i) = denom ./ var_w_j;
    ca.w.mean(:, i) = (fa.w.Mean(:, i) - var_w_j .* si.w.mu(:, i)) ./ denom;
end

ca.gamma.a = pr.rho * ones(pr.K, pr.m);
ca.gamma.b = (1 - pr.rho) * ones(pr.K, pr.m);

end


function [ti, z] = compute_prior_tilt(ca, pr)

t = ca.w.tau + 1./pr.tau2;
g_var = 1./ca.w.tau; % for gamma0
mcav2 = ca.w.mean.^2;
log_z_gamma0 = log(ca.gamma.b) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
g_var = pr.tau2 + g_var; % for gamma1
log_z_gamma1 = log(ca.gamma.a) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
z_gamma0 = exp(log_z_gamma0 - log_z_gamma1);
z_gamma1 = ones(size(log_z_gamma1));
z = 1 + z_gamma0;

ti.w.mean = z_gamma1 .* (ca.w.tau .* ca.w.mean) ./ t ./ z;
e2_w_tilt = z_gamma1 .* (1./t + 1./t.^2 .* (ca.w.tau .* ca.w.mean).^2) ./ z;
ti.w.var = e2_w_tilt - ti.w.mean.^2;

ti.gamma.mean = z_gamma1 ./ z;

end


function [si, nonpositive_cavity_vars, nonpositive_site_var_proposals] = site_updates_prior(si, ca, ti, op)

nonpositive_site_var_proposals = false;

% skip negative cavs
update_inds = ca.w.tau(:) > 0;
nonpositive_cavity_vars = ~all(update_inds);

new_tau_w_site = 1 ./ ti.w.var - ca.w.tau;

switch op.robust_updates
    case 0
    case 1
        inds_tmp = new_tau_w_site(:) > 0;
        nonpositive_site_var_proposals = ~all(inds_tmp);
        update_inds = update_inds & inds_tmp;
    case 2
        inds = new_tau_w_site(:) <= 0;
        new_tau_w_site(inds) = op.min_site_prec;
        ti.w.var(inds) = 1./(op.min_site_prec + ca.w.tau(inds));
end
new_mu_w_site = ti.w.mean ./ ti.w.var - ca.w.tau .* ca.w.mean;
si.w.tau(update_inds) = (1 - op.damp) * si.w.tau(update_inds) + op.damp * new_tau_w_site(update_inds);
si.w.mu(update_inds) = (1 - op.damp) * si.w.mu(update_inds) + op.damp * new_mu_w_site(update_inds);

% TODO: use log scale for a/b_gamma computations?
si.gamma.a(update_inds) = exp((1 - op.damp) * log(si.gamma.a(update_inds)) + op.damp * log(ti.gamma.mean(update_inds) ./ ca.gamma.a(update_inds)));
si.gamma.b(update_inds) = exp((1 - op.damp) * log(si.gamma.b(update_inds)) + op.damp * log((1 - ti.gamma.mean(update_inds)) ./ ca.gamma.b(update_inds)));

end


function fa = compute_full_approximation(si, pr)

% K x K x n
fa.x.Tau = repmat(eye(pr.K, pr.K), [1 1 pr.n]) + si.lik.x.Tau;
fa.x.Cov = zeros(size(fa.x.Tau));
% K x n
fa.x.Mu = si.lik.x.Mu;
fa.x.Mean = zeros(size(fa.x.Mu));
for i = 1:pr.n
    fa.x.Cov(:, :, i) = inv(fa.x.Tau(:, :, i));
    fa.x.Mean(:, i) = fa.x.Cov(:, :, i) * fa.x.Mu(:, i);
end


% K x K x m and K x m
fa.w.Tau = zeros(pr.K, pr.K, pr.m);
fa.w.Cov = zeros(pr.K, pr.K, pr.m);
fa.w.Mu = si.prior.w.mu + si.lik.w.Mu;
fa.w.Mean = zeros(pr.K, pr.m);

for i = 1:pr.m
    fa.w.Tau(:, :, i) = diag(si.prior.w.tau(:, i)) + si.lik.w.Tau(:, :, i);
    fa.w.Cov(:, :, i) = inv(fa.w.Tau(:, :, i));
    fa.w.Mean(:, i) = fa.w.Cov(:, :, i) * fa.w.Mu(:, i);
    
end

fa.P_gamma = si.prior.gamma.a .* pr.rho;

end


function [converged, conv] = report_progress_and_check_convergence(conv, iter, z, fa, op)

conv_z = mean(abs(z(:) - conv.z_old(:)));
conv_P_gamma = mean(abs(fa.P_gamma(:) - conv.P_gamma_old(:)));

if op.verbosity > 0 && mod(iter, op.verbosity) == 0
    fprintf(1, '%d, conv = [%.2e %.2e], damp = %.2e\n', iter, conv_z, conv_P_gamma, op.damp);
end

%converged = conv_z < op.threshold && conv_P_gamma < op.threshold;
converged = conv_P_gamma < op.threshold;

conv.z_old = z;
conv.P_gamma_old = fa.P_gamma;

end
