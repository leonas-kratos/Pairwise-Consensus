function [gt, arc] = build_ground_truth(waypoints, spacing)
  segs    = diff(waypoints, 1, 1);
  seg_len = sqrt(sum(segs.^2, 2));
  total   = sum(seg_len);

  n_pts   = max(floor(total / spacing), 2);
  cum     = [0; cumsum(seg_len)];
  q       = linspace(0, total, n_pts)';

  gt      = zeros(n_pts, 2);

  for i = 1:n_pts
    d   = q(i);
    idx = find(cum <= d, 1, 'last');
    idx = min(idx, size(segs,1));

    frac = (d - cum(idx)) / (seg_len(idx) + 1e-9);
    gt(i,:) = waypoints(idx,:) + frac * segs(idx,:);
  end

  arc = q;
end
