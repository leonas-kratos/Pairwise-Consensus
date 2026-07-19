function h = make_gauss_kernel(n_half, sigma)

  idx = (-n_half:n_half)';

  h = exp( ...
    -0.5 * idx.^2 / ...
    (sigma^2 + 1e-12));

  h = h / sum(h);

end
