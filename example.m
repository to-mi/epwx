clear all; close all;
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',123456));

%% generate some data
probit = true; % set to false to use Gaussian likelihood
N = 100;
M = 500;
sigma2 = 1;
omega = 0.1;
tau2 = 1;

x_true = randn(N, 1);
w_true = zeros(1, M);
w_true(1:(omega * M)) = sqrt(tau2) * randn(1, omega*M);
y_lat = x_true * w_true;
y = y_lat + sqrt(sigma2) * randn(N, M);

if probit
  % class labels need to be coded as -1,+1
  y = 2*(y > 0) - 1;
end

%% EP
% likelihood type for each column of y (EP code allows mixed data, others currently do not)
% 0 = Gaussian, 1 = probit
y_lik_type = ones(M, 1) * probit;
% options
opts = [];
opts.verbosity = 1; % how often to report progress (1 = every iteration, 2 = every second etc.)
opts.damp = 0.7; % EP damping
opts.damp_decay = 0.999; % decay in EP damping per iteration
opts.min_site_prec = 1e-6; % restrict minimum site precision value
opts.max_iter = 100; % maximum number of iterations
opts.threshold = 0.5e-3; % threshold for convergence checking (the algorithms use change in posterior probabilities of gamma = 1 currently for convergence monitoring)
opts.frac = 1; % fraction parameter for EP likelihood updates; does not work correctly if using probit likelihood, so use 1!
opts.robust_updates = 2; 
% prior settings
prior = [];
prior.K = 1; % latent dimensionality, i.e., the number of factors
prior.sigma2 = sigma2; % residual variance
prior.rho = omega; % sparsity parameter
prior.tau2 = tau2; % w ("loadings") prior variance

% run EP
% outputs:
% - fa_ep-struct contains the posterior approximation
% - si_ep-struct contains the site terms
fprintf('Running EP\n');
tic
[fa_ep, si_ep, converged_ep] = ep_spca_factcov(y, y_lik_type, prior, opts);
toc


%% VB-EP
% mostly the same options & prior settings and output
opts.threshold = 1e-5; % VB-EP is much faster so might as well use stricker convergence criteria
fprintf('Running VB-EP\n');
if probit
  tic
  [fa_vb, si_vb, converged_vb] = vbep_spca_probit(y, prior, opts);
  toc
else
  tic
  [fa_vb, si_vb, converged_vb] = vbep_spca_gaussian(y, prior, opts);
  toc
end


%% Gibbs
% using the VB-EP result for initialization
% prior is the same as for EP and VB-EP, but some further options need to be set
opts.nsamples = 5000; % number of samples
opts.save_y_samples = false;
opts.verbosity = 500;
opts.nwarmup = 1000; % number of warm-up samples
fprintf('Running Gibbs sampling\n');
if probit
    tic
    [fa_gibbs, samples] = gibbs_spca_probit(y, prior, opts, fa_vb);
    toc
else
    tic
    [fa_gibbs, samples] = gibbs_spca_gaussian(y, prior, opts, fa_vb);
    toc
end

%% plot EP/VB-EP against Gibbs (w mean)
figure(1); clf; hold on;
minval = min([fa_gibbs.w.Mean(:); fa_ep.w.Mean(:); fa_vb.w.Mean(:)]);
maxval = max([fa_gibbs.w.Mean(:); fa_ep.w.Mean(:); fa_vb.w.Mean(:)]);
plot([minval maxval], [minval maxval], '-', 'color', 0.5 * [1 1 1]);
h = [0 0];
% the model is unidentified w.r.t. switching the sign of w and x, so fix it
% for the comparison:
sign_ep = sign(corr(fa_gibbs.w.Mean(:), fa_ep.w.Mean(:)));
sign_vb = sign(corr(fa_gibbs.w.Mean(:), fa_ep.w.Mean(:)));
h(1) = plot(fa_gibbs.w.Mean(:), sign_ep*fa_ep.w.Mean(:), '.', 'color', [0.6 0 0]);
h(2) = plot(fa_gibbs.w.Mean(:), sign_vb*fa_vb.w.Mean(:), '.', 'color', [0 0 0.6]);
xlabel('Gibbs')
ylabel('Approx')
legend(h, 'EP', 'VB-EP', 'location', 'NorthWest');
title('Posterior mean of w');
axis equal

%% plot EP/VB-EP against Gibbs (posterior probabilities of gamma = 1)
figure(2); clf; hold on;
plot([0 1], [0 1], '-', 'color', 0.5 * [1 1 1]);
h = [0 0];
h(1) = plot(fa_gibbs.P_gamma(:), fa_ep.P_gamma(:), '.', 'color', [0.6 0 0]);
h(2) = plot(fa_gibbs.P_gamma(:), fa_vb.P_gamma(:), '.', 'color', [0 0 0.6]);
xlabel('Gibbs')
ylabel('Approx')
xlim([0 1]); ylim([0 1]);
legend(h, 'EP', 'VB-EP', 'location', 'NorthWest');
title('Posterior probabilities of gamma = 1');
axis equal
