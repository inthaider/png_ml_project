%%
%
% Import files and extract data
%

files = {'chirp.mat'
        'gong.mat'
        'handel.mat'
        'laughter.mat'
        'splat.mat'
        'train.mat'};

S = zeros(10000,6);
for i = 1:6
    test     = load(files{i});
    y        = test.y(1:10000,1);
    S(:,i)   = y;
end

% Number of 'observed mixtures' we want to provide RICA
nmix = 2;
% Number of features/latent components we want to extract
q = 6;

rng default % For reproducibility
mixdata = S*randn(q) + randn(1,q);
mixdata = mixdata(:, 1:nmix);


%%
%
% Compare the mixtures with latent comps and prewhiten
%

% Plot the source latent comps and the observed mixtures
% figure
% for i = 1:6
%     subplot(2,6,i)
%     plot(S(:,i))
%     title(['Sound ',num2str(i)])

%     subplot(2,6,i+6)
%     if i <= nmix % only plot the mixtures to be used by RICA
%         plot(mixdata(:,i))
%         title(['Mix ',num2str(i)])
%     end
% end

% Prewhiten the mixed data
mixdata = prewhiten(mixdata);


%%
%
% Optimize parameters
%

% Using the same mixed data as above
Xtrain = mixdata;

% To remove sources of variation, fix an initial transform weight matrix.
W = randn(1e2,1e2);

% Create hyperparameters for the objective function.
iterlim = optimizableVariable('iterlim',[5,1e12],'Type','integer');
lambda = optimizableVariable('lambda',[8e2,15e2]);
gradtol = optimizableVariable('gradtol',[1e-10, 1e-7]);
steptol = optimizableVariable('steptol', [1e-10, 1e-7]);
vars = [iterlim, lambda, gradtol, steptol];

% Run the optimization without the warnings that occur when the internal optimizations do not run to completion. Run for 60 iterations instead of the default 30 to give the optimization a better chance of locating a good value.
maxevals = 60;
warning('off','stats:classreg:learning:fsutils:Solver:LBFGSUnableToConverge');
results = bayesopt(@(x) filterica(x,Xtrain,W,q,nmix),vars, ...
    'UseParallel',true,'MaxObjectiveEvaluations',maxevals, ...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'IsObjectiveDeterministic', false);
warning('on','stats:classreg:learning:fsutils:Solver:LBFGSUnableToConverge');

% Extract best params:
optiterlim = results.XAtMinObjective.iterlim;
optlambda= results.XAtMinObjective.lambda;
optgradtol = results.XAtMinObjective.gradtol;
optsteptol = results.XAtMinObjective.steptol;


%%
%
% Apply RICA
%

iterlim = optiterlim;
lambda = optlambda;
gradtol = optgradtol;
steptol = optsteptol;
q = q;

% Model = rica(mixdata,q,'NonGaussianityIndicator',ones(6,1)); % create rica model
Model = rica(mixdata, q, ...
        'Standardize', true, 'NonGaussianityIndicator',ones(6,1), 'VerbosityLevel', 1, ...
        'Lambda', lambda, 'IterationLimit', iterlim, ...
        'GradientTolerance', gradtol, 'StepTolerance', steptol)
% Model = rica(mixdata,q, 'InitialTransformWeights',Model.TransformWeights, 'Lambda', lambda, 'NonGaussianityIndicator',ones(6,1), 'VerbosityLevel', 1, 'GradientTolerance',1e-9, 'StepTolerance',1e-9, 'IterationLimit',1e6)
ogunmixed = transform(Model,mixdata); % extract the unmixed comps

%
% Plot the extracted/unmixed signals
%

figure
for i = 1:6
    subplot(2,6,i)
    plot(S(:,i))
    title(['Sound ',num2str(i)])
    subplot(2,6,i+6)
    plot(ogunmixed(:,i))
    title(['Unmix ',num2str(i)])
end


%%
%
% Plot the extracted/unmixed signals in the right order
%

% Reorder the unmixed signals correctly
% unmixed = ogunmixed(:,[2, 5, 3, 6, 4, 1]);
% for i = 1:6
%     unmixed(:,i) = unmixed(:,i)/norm(unmixed(:,i))*norm(S(:,i));
% end

% figure
% for i = 1:6
%     subplot(2,6,i)
%     plot(S(:,i))
%     ylim([-1,1])
%     title(['Sound ',num2str(i)])
%     subplot(2,6,i+6)
%     plot(unmixed(:,i))
%     ylim([-1,1])
%     title(['Unmix ',num2str(i)])
% end


%%
%
% RICA Params
%
ricafit = Model.FitInfo;
ricafititer = ricafit.Iteration;
ricafitobj = ricafit.Objective;
ricafitobjmin = min(ricafitobj);

figure
plot(ricafititer, ricafitobj)

Model.TransformWeights;
% display(ricafitobj)


%%
%
% Objective function for bayesopt (optimizer for parameters)
%

% Feature extraction functions have these tuning parameters:
%
% Iteration limit
% Function, either rica or sparsefilt
% Parameter Lambda
% Number of learned features q
%
% To search among the available parameters effectively, try bayesopt. Use the following objective function, which includes parameters passed from the workspace.

function objective = filterica(x,Xtrain,winit,q,nmix)

    initW = winit(1:nmix,1:q);

    Mdl = rica(Xtrain,q,'Lambda',x.lambda,'IterationLimit',x.iterlim, ...
        'InitialTransformWeights',initW,'Standardize',true, ...
        'GradientTolerance', x.gradtol, 'StepTolerance',x.steptol);
    
    NewX = transform(Mdl,Xtrain);
    ricafit = Mdl.FitInfo;
    ricafitobj = ricafit.Objective;
    objective = min(ricafitobj);

end


%%
%
% Prewhiten function
%

function Z = prewhiten(X)
    % X = N-by-P matrix for N observations and P predictors
    % Z = N-by-P prewhitened matrix
    
        % 1. Size of X.
        [N,P] = size(X);
        assert(N >= P);
    
        % 2. SVD of covariance of X. We could also use svd(X) to proceed but N
        % can be large and so we sacrifice some accuracy for speed.
        [U,Sig] = svd(cov(X));
        Sig     = diag(Sig);
        Sig     = Sig(:)';
    
        % 3. Figure out which values of Sig are non-zero.
        tol = eps(class(X));
        idx = (Sig > max(Sig)*tol);
        assert(~all(idx == 0));
    
        % 4. Get the non-zero elements of Sig and corresponding columns of U.
        Sig = Sig(idx);
        U   = U(:,idx);
    
        % 5. Compute prewhitened data.
        mu = mean(X,1);
        Z = bsxfun(@minus,X,mu);
        Z = bsxfun(@times,Z*U,1./sqrt(Sig));
end