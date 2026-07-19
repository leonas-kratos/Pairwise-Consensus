function scores = pc_consensus_scores( ...
    innovations, d_raw, sigma, anchors, anchor_pair)

  N   = length(d_raw);
  eps = 1e-9;
  nu  = 4.0;

  residual = abs(innovations);

  tri_viol = zeros(N, 1);
  cnt      = zeros(N, 1);

  for i = 1:N

    for j = i+1:N

      D_ij = anchor_pair(i,j);

      viol = max( ...
        0, ...
        abs(d_raw(i) - d_raw(j)) - D_ij);

      tri_viol(i) = tri_viol(i) + viol;
      tri_viol(j) = tri_viol(j) + viol;

      cnt(i) = cnt(i) + 1;
      cnt(j) = cnt(j) + 1;

    end

  end

  cnt = max(cnt, 1);

  tri_viol = tri_viol ./ cnt;

  residual = residual + tri_viol;

  scores = zeros(N, 1);

  for i = 1:N

    s = 0;

    for j = 1:N

      if i == j
        continue;
      end

      diff_ij = residual(i) - residual(j);

      c = ...
        (1 + diff_ij^2 / ...
        (nu * sigma^2 + eps))^(-(nu+1)/2);

      s = s + c;

    end

    scores(i) = s / (N-1);

  end

end
