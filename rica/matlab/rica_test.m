% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA

% files = {'../input-data/matlab/z_chisq_seeds501982z_3085chi.mat'
%         '../input-data/matlab/zg_chisq_seeds501982z_3085chi.mat'
%         '../input-data/matlab/zng_chisq_seeds501982z_3085chi.mat'};
files = {'../input-data/matlab/zg_chisq_seeds501982z_3085chi.mat'
        '../input-data/matlab/zng_chisq_seeds501982z_3085chi.mat'};

n = 2;
Z = zeros(4194304, n);
for i = 1:n
    test     = load(files{i});
    y        = test.y(1, :)'; % The apostrophe denotes transpose.
    Z(:,i)   = y;
end
% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA

%% 






rng default % For reproducibility

% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA
mixzeta = Z*randn(n) + randn(1,n);

figure
figtitle = sgtitle('Source & Mixed Chi^2_e nonG Zeta');
for i = 1:n
    subplot(2,n,i)
    plot(Z(:,i))
    title(['Source Zeta ',num2str(i)])
    subplot(2,n,i+n)
    plot(mixzeta(:,i))
    title(['Unmix Zeta ',num2str(i)])
end

% saveas(gcf,'figs/plt-src_and_mix-chisq_zng.png')
exportgraphics(gcf,'figs/plt-src_and_mix-chisq_zng.png')

% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA

%% 






% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA
mixzeta = prewhiten(mixzeta);

% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA

%% 






% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA
q = n;
Model = rica(mixzeta,q, 'VerbosityLevel', 1, 'GradientTolerance',1e-7, 'StepTolerance',1e-6);
Model = rica(mixzeta,q, 'VerbosityLevel', 1, 'InitialTransformWeights',Model.TransformWeights, 'GradientTolerance',1e-10, 'StepTolerance',1e-10);
unmixedzeta = transform(Model, mixzeta);
% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA

%%






% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA
figure
sgtitle('Source vs. RICA-unmixed Chi^2_e nonG Zeta')
for i = 1:n
    subplot(2,n,i)
    plot(Z(:, i))
    title(['Source Zeta ',num2str(i)])
    subplot(2,n,i+n)
    plot(unmixedzeta(:,i))
    title(['Unmix Zeta ',num2str(i)])
end

% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA

%% 






% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA 
unmixedzeta = unmixedzeta(:,[2,1]);
for i = 1:2
    unmixedzeta(:,i) = unmixedzeta(:,i)/norm(unmixedzeta(:,i))*norm(Z(:,i));
end

figure
sgtitle('Source vs. RICA-unmixed Chi^2_e nonG Zeta')
for i = 1:n
    subplot(2,n,i)
    plot(Z(:,i))
    % ylim([-5,5])
    title(['Source Zeta ',num2str(i)])
    subplot(2,n,i+n)
    plot(unmixedzeta(:,i))
    % ylim([-5,5])
    title(['Unmix Zeta ',num2str(i)])
end

exportgraphics(gcf,'figs/plt-unmix-src_vs_rica-chisq_zng.png')
% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA

%%





 
% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA
sourceng = Z(:, 2);
ricang = unmixedzeta(:, 2);
% minsource = min(cdfsource); maxsource = max(cdfsource);
% minrica = min(cdfrica); maxrica = max(cdfrica);

% tmp = [cdfsource, cdfrica];
% tmpmax = max(tmp(:)); tmpmin = min(tmp(:));
% x = -2:.1:8;
% x = linspace(tmpmin, tmpmax);

numbins = 600;

% PLOT PDF comparison
figure
histogram(sourceng, numbins, 'Normalization','pdf', 'EdgeColor','auto', 'DisplayStyle', 'stairs'); hold on;
histogram(ricang, numbins, 'Normalization','pdf', 'EdgeColor','auto', 'DisplayStyle', 'stairs');
grid on;
title('PDFs - Source vs. RICA-unmixed Chi^2_e nonG comps')
legend('Original Chi_e^2 nonG', 'RICA-unmixed Chi_e^2 nonG', 'Location', 'best');
hold off;

exportgraphics(gcf,'figs/pdf-chisq_ng.png')

% PLOT CDF comparison
figure
ecdf(sourceng, 'Bounds', 'on', 'Alpha', 0.3); hold on;
ecdf(ricang, 'Bounds', 'on', 'Alpha', 0.3);
grid on;
title('Empirical CDFs: Source vs. RICA-unmixed Chi^2_e nonG comps')
legend('Original Chi_e^2 nonG', 'Source Lower Confidence','Source Upper Confidence', 'RICA-unmixed Chi_e^2 nonG', 'RICA Lower Confidence', 'RICA Upper Confidence', 'Location', 'best');
hold off;

exportgraphics(gcf,'figs/ecdf-src_vs_rica-chisq_zng.png')

[h,p,k] = kstest2(Z(:, 2),unmixedzeta(:, 2),'Alpha',0.05);
% ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA % ZETA


%%






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