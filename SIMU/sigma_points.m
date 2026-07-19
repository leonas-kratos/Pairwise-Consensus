function pts = sigma_points(x, P, c)

  n = length(x);

  try
    S = chol(c * P, 'lower');

  catch
    S = chol(c * (P + eye(n) * 1e-6), 'lower');

  end

  pts = zeros(2*n+1, n);

  pts(1,:) = x';

  for i = 1:n

    pts(i+1,:)   = (x + S(:,i))';
    pts(n+i+1,:) = (x - S(:,i))';

  end

end
