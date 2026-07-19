function [Wm, Wc, c] = ukf_weights(n, alpha, beta, kappa)

  lam = alpha^2 * (n + kappa) - n;

  c = n + lam;

  Wm = ones(2*n+1, 1) * 0.5 / c;
  Wc = ones(2*n+1, 1) * 0.5 / c;

  Wm(1) = lam / c;

  Wc(1) = lam / c + (1 - alpha^2 + beta);

end
