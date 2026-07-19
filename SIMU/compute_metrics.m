function m = compute_metrics(e)

  e = sort(e(:));

  m.rmse = sqrt(mean(e.^2));
  m.mae  = mean(e);

  m.cep50 = percentile_local(e, 50);
  m.cep90 = percentile_local(e, 90);
  m.p95   = percentile_local(e, 95);

  m.mx = max(e);

end


function p = percentile_local(x, q)

  n = length(x);

  pos = 1 + (n - 1) * q / 100;

  lo = floor(pos);
  hi = ceil(pos);

  if lo == hi

    p = x(lo);

  else

    p = x(lo) + ...
        (pos - lo) * (x(hi) - x(lo));

  end

end
