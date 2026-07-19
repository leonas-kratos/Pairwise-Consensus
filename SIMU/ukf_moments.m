function [z_hat, Pzz, Pxz] = ukf_moments( ...
    x_pred, P_pred, Wm, Wc, c, anchors, H)

  n = length(x_pred);

  pts = sigma_points(x_pred, P_pred, c);

  M = size(anchors, 1);

  Zpts = zeros(2*n+1, M);

  for i = 1:2*n+1

    Zpts(i,:) = h_obs( ...
      pts(i,:)', anchors, H)';

  end

  z_hat = Wm' * Zpts;

  Pzz = zeros(M, M);
  Pxz = zeros(n, M);

  for i = 1:2*n+1

    dz = Zpts(i,:)' - z_hat';
    dx = pts(i,:)'  - x_pred;

    Pzz = Pzz + Wc(i) * (dz * dz');
    Pxz = Pxz + Wc(i) * (dx * dz');

  end

end
