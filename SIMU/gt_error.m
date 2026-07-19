function e = gt_error(pos, gt)

  e = zeros(size(pos,1), 1);

  for i = 1:size(pos,1)

    dists = sqrt( ...
      sum((gt - pos(i,:)).^2, 2));

    e(i) = min(dists);

  end

end
