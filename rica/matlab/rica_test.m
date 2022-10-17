files = {'chirp.mat'
        'gong.mat'
       Opening log file:  /home/haider/java.log.10964
plat.mat'
        'train.mat'};

file = {'z_chisq_seeds501982z_3085chi.mat'};


S = zeros(10000,6);
for i = 1:6
    test     = load(files{i});
    y        = test.y(1:10000,1);
    S(:,i)   = y;
end

sourceload = load(file{1});
source = sourceload.y;


%% 


rng default % For reproducibility
mixdata = S*randn(6) + randn(1,6);

figure
for i = 1:6
    subplot(2,6,i)
    plot(S(:,i))
    title(['Sound ',num2str(i)])
    subplot(2,6,i+6)
    plot(mixdata(:,i))
    title(['Mix ',num2str(i)])
end




mixdata = prewhiten(mixdata);

figure
histogram(S(:,1))

q = 6;
Mdl = rica(mixdata,q,'NonGaussianityIndicator',ones(6,1));

unmixed = transform(Mdl,mixdata);

figure
for i = 1:6
    subplot(2,6,i)
    plot(S(:,i))
    title(['Sound ',num2str(i)])
    subplot(2,6,i+6)
    plot(unmixed(:,i))
    title(['Unmix ',num2str(i)])
end

unmixed = unmixed(:,[2,5,4,6,3,1]);
for i = 1:6
    unmixed(:,i) = unmixed(:,i)/norm(unmixed(:,i))*norm(S(:,i));
end

figure
for i = 1:6
    subplot(2,6,i)
    plot(S(:,i))
    ylim([-1,1])
    title(['Sound ',num2str(i)])
    subplot(2,6,i+6)
    plot(unmixed(:,i))
    ylim([-1,1])
    title(['Unmix ',num2str(i)])
end


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