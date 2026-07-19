function P = make_spd(P)

  P = 0.5 * (P + P');

  P = P + eye(size(P,1)) * 1e-9;

end
