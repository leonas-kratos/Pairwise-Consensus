function results = run_all_filters(d_meas, anchors, H, gt_xy, ...
    UKF_Q, UKF_R, ...
    HUBER_Q, HUBER_R, HUBER_DELTA, HUBER_ITER, ...
    MCC_Q, MCC_R, MCC_BW, MCC_ITER, ...
    PC_Q, PC_RBASE, PC_RSCALE, PC_SIGMA, ...
    GUKF_Q, GUKF_R, GUKF_SIGMA, GUKF_NHALF, ...
    UKF_ALPHA, UKF_BETA, UKF_KAPPA, ANCHOR_PAIR)

  T = size(d_meas, 1);
  N = size(anchors, 1);
  n = 2;

  [Wm, Wc, c_ukf] = ukf_weights(n, UKF_ALPHA, UKF_BETA, UKF_KAPPA);

  x0 = ls_position(d_meas(1,:)', anchors);
  if any(isnan(x0)); x0 = [4400; 2000]; end

  pos_raw   = zeros(T, 2);
  pos_ukf   = zeros(T, 2);
  pos_huber = zeros(T, 2);
  pos_mcc   = zeros(T, 2);
  pos_pc    = zeros(T, 2);
  pos_gukf  = zeros(T, 2);

  x_ukf = x0; P_ukf = eye(n)*1e6;
  x_hub = x0; P_hub = eye(n)*1e6;
  x_mcc = x0; P_mcc = eye(n)*1e6;
  x_pc  = x0; P_pc  = eye(n)*1e6;
  pc_kf_x = d_meas(1,:)';
  pc_kf_P = ones(N, 1);
  pc_kf_Q = 0.01; pc_kf_R = 200.0;
  x_gukf = x0; P_gukf = eye(n)*1e6;
  gukf_buf = [];
  gukf_ker = make_gauss_kernel(GUKF_NHALF, GUKF_SIGMA);
  gukf_win = 2*GUKF_NHALF + 1;

  Q_ukf   = eye(n) * UKF_Q;
  Q_hub   = eye(n) * HUBER_Q;
  Q_mcc   = eye(n) * MCC_Q;
  Q_pc    = eye(n) * PC_Q;
  Q_gukf  = eye(n) * GUKF_Q;

  for t = 1:T
    z = d_meas(t,:)';

    % ── Raw + LS ─────────────────────────────────────────────────────
    pos_raw(t,:) = ls_position(z, anchors)';

    % ── 1. Standard UKF ──────────────────────────────────────────────
    P_pred = P_ukf + Q_ukf;
    [z_hat, Pzz, Pxz] = ukf_moments(x_ukf, P_pred, Wm, Wc, c_ukf, anchors, H);
    R = eye(N) * UKF_R;
    Pzz_eff = Pzz + R;
    K = Pxz / Pzz_eff;
    x_ukf = x_ukf + K * (z - z_hat');
    P_ukf = make_spd(P_pred - K * Pzz_eff * K');
    pos_ukf(t,:) = x_ukf';

    % ── 2. Huber-UKF ─────────────────────────────────────────────────
    P_pred = P_hub + Q_hub;
    [z_hat, Pzz, Pxz] = ukf_moments(x_hub, P_pred, Wm, Wc, c_ukf, anchors, H);
    R_eff = eye(N) * HUBER_R;
    for iter = 1:HUBER_ITER
      Pzz_eff  = Pzz + R_eff;
      innov    = z - z_hat';
      pzz_diag = max(diag(Pzz_eff), 1e-9);
      r_scaled = innov ./ sqrt(pzz_diag);
      hub_w    = ones(N, 1);
      idx_out  = abs(r_scaled) > HUBER_DELTA;
      hub_w(idx_out) = HUBER_DELTA ./ (abs(r_scaled(idx_out)) + 1e-9);
      hub_w    = max(hub_w, 1e-4);
      R_eff    = diag(HUBER_R ./ hub_w);
    end
    Pzz_eff = Pzz + R_eff;
    K = Pxz / Pzz_eff;
    x_hub = x_hub + K * (z - z_hat');
    P_hub = make_spd(P_pred - K * Pzz_eff * K');
    pos_huber(t,:) = x_hub';

    % ── 3. MCC-UKF ───────────────────────────────────────────────────
    P_pred = P_mcc + Q_mcc;
    [z_hat, Pzz, Pxz] = ukf_moments(x_mcc, P_pred, Wm, Wc, c_ukf, anchors, H);
    x_cur = x_mcc;
    K     = zeros(n, N);
    R_eff = eye(N) * MCC_R;
    for iter = 1:MCC_ITER
      innov   = z - z_hat';
      kern_w  = exp(-0.5 * innov.^2 / (MCC_BW^2 + 1e-9));
      kern_w  = max(kern_w, 1e-4);
      R_eff   = diag(MCC_R ./ kern_w);
      Pzz_eff = Pzz + R_eff;
      K       = Pxz / Pzz_eff;
      x_new   = x_mcc + K * innov;
      if norm(x_new - x_cur) < 1e-3
        x_cur = x_new; break;
      end
      x_cur = x_new;
    end
    x_mcc = x_cur;
    P_mcc = make_spd(P_pred - K * (Pzz + R_eff) * K');
    pos_mcc(t,:) = x_mcc';

    % ── 4. PC-UKF ────────────────────────────────────────────────────
    pc_innov = zeros(N, 1);
    for i = 1:N
      P_pc_pred  = pc_kf_P(i) + pc_kf_Q;
      inn_i      = z(i) - pc_kf_x(i);
      K_i        = P_pc_pred / (P_pc_pred + pc_kf_R);
      pc_kf_x(i) = pc_kf_x(i) + K_i * inn_i;
      pc_kf_P(i) = (1 - K_i) * P_pc_pred;
      pc_innov(i) = inn_i;
    end
    scores  = pc_consensus_scores(pc_innov, z, PC_SIGMA, anchors, ANCHOR_PAIR);
    R_diag  = PC_RBASE * (1 + PC_RSCALE * (1 - scores));

    P_pred = P_pc + Q_pc;
    [z_hat, Pzz, Pxz] = ukf_moments(x_pc, P_pred, Wm, Wc, c_ukf, anchors, H);
    R_eff   = diag(R_diag);
    Pzz_eff = Pzz + R_eff;
    K = Pxz / Pzz_eff;
    x_pc = x_pc + K * (z - z_hat');
    P_pc = make_spd(P_pred - K * Pzz_eff * K');
    pos_pc(t,:) = x_pc';

    % ── 5. GUKF ──────────────────────────────────────────────────────
    gukf_buf = [gukf_buf; z'];
    if size(gukf_buf, 1) > gukf_win
      gukf_buf = gukf_buf(end-gukf_win+1:end,:);
    end
    L = size(gukf_buf, 1);
    if L < gukf_win
      h_cut = gukf_ker(gukf_win-L+1:end);
      h_cut = h_cut / sum(h_cut);
      z_smooth = (h_cut' * gukf_buf)';
    else
      z_smooth = (gukf_ker' * gukf_buf)';
    end

    P_pred = P_gukf + Q_gukf;
    [z_hat, Pzz, Pxz] = ukf_moments(x_gukf, P_pred, Wm, Wc, c_ukf, anchors, H);
    R_eff   = eye(N) * GUKF_R;
    Pzz_eff = Pzz + R_eff;
    K = Pxz / Pzz_eff;
    x_gukf = x_gukf + K * (z_smooth - z_hat');
    P_gukf = make_spd(P_pred - K * Pzz_eff * K');
    pos_gukf(t,:) = x_gukf';
  end

  results.pos_raw   = pos_raw;
  results.pos_ukf   = pos_ukf;
  results.pos_huber = pos_huber;
  results.pos_mcc   = pos_mcc;
  results.pos_pc    = pos_pc;
  results.pos_gukf  = pos_gukf;

  results.err_raw   = gt_error(pos_raw,   gt_xy);
  results.err_ukf   = gt_error(pos_ukf,   gt_xy);
  results.err_huber = gt_error(pos_huber, gt_xy);
  results.err_mcc   = gt_error(pos_mcc,   gt_xy);
  results.err_pc    = gt_error(pos_pc,    gt_xy);
  results.err_gukf  = gt_error(pos_gukf,  gt_xy);
end

