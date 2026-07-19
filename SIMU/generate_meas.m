function d_meas = generate_meas(true_pos, ANCHORS, ANCHOR_HEIGHT, ...
                                 gauss_std, nlos_prob, nlos_lambda, scenario)

  T = size(true_pos, 1);
  N = size(ANCHORS, 1);

  d_meas = zeros(T, N);

  for t = 1:T
    for i = 1:N

      dx = true_pos(t,1) - ANCHORS(i,1);
      dy = true_pos(t,2) - ANCHORS(i,2);

      d_true = sqrt(dx^2 + dy^2);

      noise = gauss_std * randn();

      if strcmp(scenario, 'NLOS') && rand() < nlos_prob

        bias = -log(rand()) * nlos_lambda;
        bias = abs(bias);

        d_meas(t,i) = max(d_true + bias + noise, 1.0);

      else

        d_meas(t,i) = max(d_true + noise, 1.0);

      end

    end
  end

end
