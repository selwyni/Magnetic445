using DataFrames, DataFramesMeta, CSV, Gadfly, Lazy, Optim
landmarks = DataFrame(Lat = [42.7, 42.8],
                      Lon = [-122.2, -122.54])

data = CSV.read("/Users/selwyni/Desktop/CMU/JuliaWorkbook/sesame.csv")

function dist(phi1::Real, lambda1::Real, phi2::Real, lambda2::Real, R)
  phi1 = phi1 * pi / 180
  lambda1 = lambda1 * pi / 180
  phi2 = phi2 * pi / 180
  lambda2 = lambda2 * pi / 180

  dphi = phi2 - phi1
  dlambda = lambda2 - lambda1
  mphi = (phi1 + phi2) / 2
  d = R * sqrt(dphi^2 + (cos(mphi) * dlambda)^2)
  return(d)
end

test = @> begin
  data
  @transform(sex = CategoricalArray(:sex))
  @select(:sex)
  first(5)
end
Gadfly.with_theme(:dark) do
  plot(data, x = :sex, y = :age, Geom.boxplot,
  Guide.title("Testboxplot"),
  )
end
