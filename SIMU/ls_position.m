function pos = ls_position(distances, anchors)

  x0 = anchors(1,1);
  y0 = anchors(1,2);

  d0 = max(distances(1), 1.0);

  N = size(anchors, 1);

  A = zeros(N-1, 2);
  b = zeros(N-1, 1);

  for i = 2:N

    xi = anchors(i,1);
    yi = anchors(i,2);

    di = max(distances(i), 1.0);

    A(i-1,:) = [
      2*(xi-x0), ...
      2*(yi-y0)
    ];

    b(i-1) = ...
      (d0^2 - di^2) ...
      - (x0^2 - xi^2) ...
      - (y0^2 - yi^2);

  end

  pos = (A' * A) \ (A' * b);

end
