function h = h_obs(state, anchors, H)

  x = state(1);
  y = state(2);

  N = size(anchors, 1);

  h = zeros(N, 1);

  for i = 1:N

    dx = x - anchors(i,1);
    dy = y - anchors(i,2);

    h(i) = sqrt(dx^2 + dy^2 + H^2);

  end

end
