% ======================================================================
%  UWB Indoor Positioning Simulation — Octave
%  V19 port: Raw+LS, UKF, Huber-UKF, MCC-UKF, PC-UKF, GUKF
%  Layout: 8800 x 4000 mm (nằm ngang)
%  Noise: Gaussian (LOS) + NLOS (Gaussian + exponential bias dương)
% ======================================================================
clear; clc; close all;
pkg load statistics;   % cho chi2inv nếu cần

% ─── SEED ─────────────────────────────────────────────────────────────
rand('state', 42);
randn('state', 42);

% ======================================================================
%  CONFIG
% ======================================================================
% Anchor positions [x, y] mm
ANCHORS = [8800, 0;
           8800, 4000;
           0,    4000;
           0,    0];
ANCHOR_HEIGHT = 1400.0;  % mm
N_ANCHORS = 4;

% Waypoints trajectory
WAYPOINTS = [400,  3200;
             400,  1000;
             8000, 1000;
             8000, 3200;
             400,  3200];

SPEED      = 200.0;   % mm/s
GT_SPACING = 5.0;     % mm

% ── Noise scenarios ───────────────────────────────────────────────────
GAUSS_STD   = 150.0;    % mm — Gaussian noise std (cả LOS lẫn NLOS)
NLOS_PROB   = 0.5;     % xác suất 1 anchor bị NLOS tại timestep t
NLOS_LAMBDA = 300.0;   % mm — exponential bias mean (bias = exprnd(lambda))
                       % bias luôn dương: d_nlos = d_true + bias + gauss

% ── UKF params chung ──────────────────────────────────────────────────
UKF_ALPHA = 1e-3;
UKF_BETA  = 2.0;
UKF_KAPPA = 0.0;

% ── Params từng filter ────────────────────────────────────────────────
UKF_Q    = 0.001;   UKF_R    = 50.0;
HUBER_Q  = 0.001;   HUBER_R  = 50.0;   HUBER_DELTA = 20.0;  HUBER_ITER = 5;
MCC_Q    = 0.001;   MCC_R    = 100.0;  MCC_BW      = 1000.0; MCC_ITER  = 5;
PC_Q     = 0.01;   PC_RBASE = 25.0;   PC_RSCALE   = 5.0;  PC_SIGMA  = 200.0;
GUKF_Q   = 0.001;   GUKF_R   = 20.0;  GUKF_SIGMA  = 5.0;   GUKF_NHALF = 1;

% ── Pre-compute anchor pair distances (triangle violation) ─────────────
ANCHOR_PAIR = zeros(N_ANCHORS, N_ANCHORS);
for i = 1:N_ANCHORS
  for j = 1:N_ANCHORS
    dx = ANCHORS(i,1) - ANCHORS(j,1);
    dy = ANCHORS(i,2) - ANCHORS(j,2);
    ANCHOR_PAIR(i,j) = sqrt(dx^2 + dy^2);
  end
end

% ======================================================================
%  BUILD GROUND TRUTH
% ======================================================================
[gt_xy, ~] = build_ground_truth(WAYPOINTS, GT_SPACING);
N_GT = size(gt_xy, 1);
expected_path = sum(sqrt(sum(diff(WAYPOINTS,1,1).^2, 2)));

% ======================================================================
%  GENERATE TRAJECTORY (ground truth positions theo thời gian)
% ======================================================================
segs    = diff(WAYPOINTS, 1, 1);
seg_len = sqrt(sum(segs.^2, 2));
total   = sum(seg_len);
T       = floor(total / SPEED * 60) + 1;   % ~60 Hz
t_vec   = linspace(0, total/SPEED, T)';
d_vec   = t_vec * SPEED;                    % distance along path

true_pos = zeros(T, 2);
cum = [0; cumsum(seg_len)];
for t = 1:T
  d   = mod(d_vec(t), total);
  idx = find(cum <= d, 1, 'last');
  idx = min(idx, size(segs,1));
  frac = (d - cum(idx)) / (seg_len(idx) + 1e-9);
  true_pos(t,:) = WAYPOINTS(idx,:) + frac * segs(idx,:);
end

% ======================================================================
%  GENERATE MEASUREMENTS — 2 scenarios
% ======================================================================
d_los  = generate_meas(true_pos, ANCHORS, ANCHOR_HEIGHT, ...
                        GAUSS_STD, NLOS_PROB, NLOS_LAMBDA, 'LOS');
d_nlos = generate_meas(true_pos, ANCHORS, ANCHOR_HEIGHT, ...
                        GAUSS_STD, NLOS_PROB, NLOS_LAMBDA, 'NLOS');

% ======================================================================
%  RUN BOTH SCENARIOS
% ======================================================================
fprintf('\n====== SCENARIO 1: LOS (Gaussian only) ======\n');
r_los = run_all_filters(d_los, ANCHORS, ANCHOR_HEIGHT, gt_xy, ...
  UKF_Q, UKF_R, HUBER_Q, HUBER_R, HUBER_DELTA, HUBER_ITER, ...
  MCC_Q, MCC_R, MCC_BW, MCC_ITER, ...
  PC_Q, PC_RBASE, PC_RSCALE, PC_SIGMA, ...
  GUKF_Q, GUKF_R, GUKF_SIGMA, GUKF_NHALF, ...
  UKF_ALPHA, UKF_BETA, UKF_KAPPA, ANCHOR_PAIR);

print_metrics('Raw+LS',    compute_metrics(r_los.err_raw));
print_metrics('UKF',       compute_metrics(r_los.err_ukf));
print_metrics('Huber-UKF', compute_metrics(r_los.err_huber));
print_metrics('MCC-UKF',   compute_metrics(r_los.err_mcc));
print_metrics('PC-UKF',    compute_metrics(r_los.err_pc));
print_metrics('GUKF',      compute_metrics(r_los.err_gukf));

fprintf('\n====== SCENARIO 2: NLOS (Gaussian + Exponential bias) ======\n');
fprintf('  NLOS prob=%.1f  bias~Exp(%.0fmm)\n', NLOS_PROB, NLOS_LAMBDA);
r_nlos = run_all_filters(d_nlos, ANCHORS, ANCHOR_HEIGHT, gt_xy, ...
  UKF_Q, UKF_R, HUBER_Q, HUBER_R, HUBER_DELTA, HUBER_ITER, ...
  MCC_Q, MCC_R, MCC_BW, MCC_ITER, ...
  PC_Q, PC_RBASE, PC_RSCALE, PC_SIGMA, ...
  GUKF_Q, GUKF_R, GUKF_SIGMA, GUKF_NHALF, ...
  UKF_ALPHA, UKF_BETA, UKF_KAPPA, ANCHOR_PAIR);

print_metrics('Raw+LS',    compute_metrics(r_nlos.err_raw));
print_metrics('UKF',       compute_metrics(r_nlos.err_ukf));
print_metrics('Huber-UKF', compute_metrics(r_nlos.err_huber));
print_metrics('MCC-UKF',   compute_metrics(r_nlos.err_mcc));
print_metrics('PC-UKF',    compute_metrics(r_nlos.err_pc));
print_metrics('GUKF',      compute_metrics(r_nlos.err_gukf));

% ======================================================================
%  PLOTS
% ======================================================================
methods   = {'Raw+LS','UKF','Huber-UKF','MCC-UKF','PC-UKF','GUKF'};
clr_rgb = [0.62 0.62 0.62;
           0.00 0.74 0.83;
           0.91 0.12 0.39;
           1.00 0.60 0.00;
           0.61 0.15 0.69;
           0.30 0.69 0.31];

% ── Figure 1: Trajectory comparison ────────────────────────────────────
figure('Name','Trajectories','Position',[50 50 1400 600]);

scenario_names = {'LOS','NLOS'};
results_list   = {r_los, r_nlos};
pos_fields     = {'pos_raw','pos_ukf','pos_huber','pos_mcc','pos_pc','pos_gukf'};

for sc = 1:2
  subplot(1, 2, sc);
  hold on;
  r = results_list{sc};

  plot(gt_xy(:,1), gt_xy(:,2), 'k--', 'LineWidth', 2, 'DisplayName', 'Ground Truth');

  for m = 1:length(methods)
      pos = r.(pos_fields{m});

      switch m
        case 1
          err = r.err_raw;
        case 2
          err = r.err_ukf;
        case 3
          err = r.err_huber;
        case 4
          err = r.err_mcc;
        case 5
          err = r.err_pc;
        case 6
          err = r.err_gukf;
      end

      rmse_v = sqrt(mean(err.^2));

      plot(pos(:,1), pos(:,2), 'Color', clr_rgb(m,:), ...
          'LineWidth', 1.2, ...
          'DisplayName', sprintf('%s (%.0fmm)', methods{m}, rmse_v));
  end

  for w = 1:4
    plot(WAYPOINTS(w,1), WAYPOINTS(w,2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    text(WAYPOINTS(w,1)+100, WAYPOINTS(w,2)+100, char('A'+w-1), 'FontSize', 12, 'FontWeight', 'bold');
  end
  for a = 1:4
    plot(ANCHORS(a,1), ANCHORS(a,2), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    text(ANCHORS(a,1)+100, ANCHORS(a,2)-200, sprintf('A%d',a), 'Color','r','FontSize',9);
  end

  title(sprintf('Trajectory — %s', scenario_names{sc}), 'FontSize', 13, 'FontWeight', 'bold');
  xlabel('X (mm)'); ylabel('Y (mm)');
  legend('Location','best','FontSize',8);
  axis equal; grid on;
  hold off;
end
saveas(gcf, 'trajectories.png');
fprintf('[✓] trajectories.png\n');

% ── Figure 2: CDF comparison ────────────────────────────────────────────
figure('Name','CDF','Position',[50 50 1400 550]);
err_fields = {'err_raw','err_ukf','err_huber','err_mcc','err_pc','err_gukf'};

for sc = 1:2
  subplot(1, 2, sc);
  hold on;
  r = results_list{sc};
  err_fields = {
    'err_raw',
    'err_ukf',
    'err_huber',
    'err_mcc',
    'err_pc',
    'err_gukf'
  };
  for m = 1:length(methods)
    e    = sort(r.(err_fields{m}));
    cdf  = (1:length(e))' / length(e);
    rmse_v = sqrt(mean(e.^2));
    lw = 2.0;
    if strcmp(methods{m}, 'Raw+LS'); lw = 1.2; end
    plot(e, cdf*100, 'Color', clr_rgb(m,:), 'LineWidth', lw, ...
         'DisplayName', sprintf('%s RMSE=%.0fmm', methods{m}, rmse_v));
  end
  xlim([0 1000]); ylim([0 100]);
  xlabel('Position Error (mm)'); ylabel('CDF (%)');
  title(sprintf('CDF — %s', scenario_names{sc}), 'FontSize', 13, 'FontWeight', 'bold');
  legend('Location','southeast','FontSize',8);
  grid on; hold off;
end
saveas(gcf, 'cdf.png');
fprintf('[✓] cdf.png\n');

% ── Figure 3: Bar chart RMSE ────────────────────────────────────────────
figure('Name','RMSE Bar','Position',[50 50 900 500]);
rmse_los  = zeros(length(methods), 1);
rmse_nlos = zeros(length(methods), 1);
for m = 1:length(methods)
  rmse_los(m)  = sqrt(mean(r_los.(err_fields{m}).^2));
  rmse_nlos(m) = sqrt(mean(r_nlos.(err_fields{m}).^2));
end
x_pos = 1:length(methods);
b1 = bar(x_pos - 0.2, rmse_los,  0.35, 'FaceColor', [0.3 0.6 0.9]);
hold on;
b2 = bar(x_pos + 0.2, rmse_nlos, 0.35, 'FaceColor', [0.9 0.4 0.3]);
for m = 1:length(methods)
  text(m-0.2, rmse_los(m)+2,  sprintf('%.0f',rmse_los(m)),  'HorizontalAlignment','center','FontSize',9);
  text(m+0.2, rmse_nlos(m)+2, sprintf('%.0f',rmse_nlos(m)), 'HorizontalAlignment','center','FontSize',9);
end
set(gca, 'XTick', x_pos, 'XTickLabel', methods, 'FontSize', 10);
ylabel('RMSE (mm)'); title('RMSE — LOS vs NLOS', 'FontSize', 13, 'FontWeight', 'bold');
legend([b1 b2], {'LOS','NLOS'}, 'Location', 'northwest');
grid on; hold off;
saveas(gcf, 'bar_rmse.png');
fprintf('[✓] bar_rmse.png\n');

fprintf('\nDone. Thay đổi params ở phần CONFIG đầu file để thử các TH khác.\n');