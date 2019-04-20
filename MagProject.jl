using DataFrames, DataFramesMeta, Lazy, Gadfly, CSV, Optim
using Statistics, KernelEstimator

###############
# Data Loading
###############

BS = CSV.read("BS4.csv", header = [:Dd, :Ex])
Dig = CSV.read("DigitizedData.csv", header = [:Dd, :Ex])
Chen = DataFrame(D = [2.24, 2.482, 2.5, 2.49, 3.66, 2.73, 2.65, 2.69, 2.75, 2.74, 2.68, 2.714, 2.78],
                 d = [1.71, 1.58, 1.38, 1.27, 1.71, 2.94, 2.33, 2.11, 1.93, 3.44, 2.72, 2.47, 2.25],
                 Metal = CategoricalArray(["Mn", "Fe", "Co", "Ni", "Cu2MnAl", "Mo", "Ru" , "Rh", "Pd", "W", "Os", "Ir", "Pt"]))
BS = @> begin
  BS
  @transform(Metal = CategoricalArray(["Mn", "Fe", "Co", "Ni"]))
end
Chen = @> begin
  Chen
  @transform(Dd = :D ./ :d)
end

Sval = DataFrame(Metal = ["Mn", "Fe", "Co", "Ni"],
          S = [2.5, 2, 1.5, 1])
a = BS[:, :Ex]
b = BS[:, :Dd]

Sval = @> begin
  Sval
  @transform(SS1 = :S .* (:S .+ 1),
             Curie = [missing, 1043, 1400, 627],
             Dd = b,
             Min = b .- 0.05,
             Max = b ,
             kB = [1.38, 1.38, 1.38, 1.38],
             JexKnown = a)
end

bulkmod = DataFrame(Metal = ["Mn", "Fe", "Co", "Ni"],
                    BM = [1.2e11, 1.7e11, 1.8e11, 1.8e11],
                    a0 = [2.587, 2.867, 3.54, 3.499],
                    D1 = BS[:, :Dd] .* Chen[1:4, :d],
                    xtal = ["BCC", "BCC", "FCC", "FCC"],
                    d = Chen[1:4, :d])


bulkmod = @transform(bulkmod, a0 = :D1 * 2/ sqrt(3))
@byrow! bulkmod begin
  if :xtal == "BCC"
    :a0 = :D1 * 2 / sqrt(3)
  else
    :a0 = :D1 * 2 / sqrt(2)
  end
end

##############################
# Modeling Functions
##############################

function fixmis(x)
  collect(skipmissing(x))
end

function rose_model(x, p)
  (p[1] .+ p[2] .* x .+ p[3] .* x.^2 .+ p[4] .* x.^3) .* map(exp, (p[5] .* x))
end

function poly(x, coef)
  rsum = zeros(length(x))
  deg = length(coef) - 1
  for i in 0:deg
    rsum .+= coef[i+1] .* x.^i
  end
  return(rsum)
end

function model_fit(p, model, data)
  x = collect(skipmissing(data[:, :Dd]))
  y = collect(skipmissing(data[:, :Ex]))
  yhat = model(x, p)
  mse = mean((yhat - y).^2)
  return(mse)
end

function rose_model_shift_mse(shift, data)
  yhat = rose_model_shift(shift, fixmis(data[:, :Dd]), rose_params)
  mean((yhat .- fixmis(data[:, :Ex])).^2)
end

function rose_model_shift(shift, x, params)
  rose_model(fixmis(x) .+ shift[1], params) .+ shift[2]
end

function volRange(a0, BM, dP, endP)
  initP = 1e5
  niter::Int = (endP - initP) / dP
  df = DataFrame(iter = Int32[], P = Float64[], V = Float64[], a0 = Float64[])
  push!(df, [0, initP, a0^3, a0])
  for iter in 1:niter
    prevV = df[iter, :V]
    prevP = df[iter, :P]
    newV = prevV - (prevV * dP / BM)
    newP = prevP + dP
    newa0 = cbrt(newV)
    push!(df, [iter, newP, newV, newa0])
  end
  df
end

function ExVals(bulkmodRow, xtal, rose_params)
  df = volRange(bulkmod[bulkmodRow, :a0], bulkmod[bulkmodRow, :BM], 1e5, 1e10)
  df = @transform(df, D = ifelse(xtal == "BCC", :a0 * sqrt(3) / 2, :a0 * sqrt(2) / 2))
  df2 = DataFrame(Dd = df[:, :D] ./ bulkmod[bulkmodRow, :d])
  df2 = @transform(df2, Ex = rose_model(:Dd, rose_params))
  df2[range(1, nrow())]
end

##############################
# Optimizations
##############################

rose = optimize((d -> model_fit(d, rose_model, Dig)), zeros(5),  BFGS())
rose_params = Optim.minimizer(rose)
poly2val = optimize((d -> model_fit(d, poly, Dig)), zeros(3), BFGS())
poly2_params =  Optim.minimizer(poly2val)
poly3val = optimize((d -> model_fit(d, poly, Dig)), zeros(4), BFGS())
poly3_params =  Optim.minimizer(poly3val)
poly4val = optimize((d -> model_fit(d, poly, Dig)), zeros(5), BFGS())
poly4_params = Optim.minimizer(poly4val)
nprval = npr(fixmis(Dig[:, :Dd]), fixmis(Dig[:, :Ex]), xeval = fixmis(Dig[:, :Dd]))

roseFeCo = optimize((d -> model_fit(d, rose_model, BS[2:3, :])), zeros(5), BFGS())
roseFeCo_params = Optim.minimizer(roseFeCo)
roseFerro = optimize((d -> model_fit(d, rose_model, BS[2:4, :])), zeros(5), BFGS())
roseFerro_params = Optim.minimizer(roseFerro)
roseAll = optimize((d -> model_fit(d, rose_model, BS)), zeros(5), BFGS())
roseAll_params = Optim.minimizer(roseAll)

rose4 = optimize((d -> rose_model_shift_mse(d, BS)), zeros(2),  BFGS())
rose4_shift = Optim.minimizer(rose4)
rose4_layer = layer(x = Dig[:, :Dd], y = rose_model_shift(rose4_shift, Dig[:,:Dd], rose_params),
                    Geom.line, Theme(default_color = "red"))
rose4_mse = rose_model_shift_mse(rose4_shift, BS)

valferro = optimize((d -> rose_model_shift_mse(d, BS[2:4,:])), zeros(2),  BFGS())
roseferro_shift = Optim.minimizer(valferro)
roseferro_layer = layer(x = Dig[:, :Dd], y = rose_model_shift(roseferro_shift, Dig[:,:Dd], rose_params),
                    Geom.line, Theme(default_color = "blue"))
roseferro_mse = rose_model_shift_mse(roseferro_shift, BS)

valFeCo = optimize((d -> rose_model_shift_mse(d, BS[2:3,:])), zeros(2),  BFGS())
roseFeCo_shift = Optim.minimizer(valFeCo)
roseFeCo_layer = layer(x = Dig[:, :Dd], y = rose_model_shift(roseFeCo_shift, Dig[:,:Dd], rose_params),
                    Geom.line, Theme(default_color = "orange"))
roseFeCo_mse = rose_model_shift_mse(roseFeCo_shift, BS)


pm_Mn = DataFrame(Dd = range(Sval[1, :Min], Sval[1, :Max], length = 200),
                  JexPred = rose_model(range(Sval[1, :Min], Sval[1, :Max], length = 200), rose_params))

pm_Fe = DataFrame(Dd = range(Sval[2, :Min], Sval[2, :Max], length = 200),
                  JexPred = rose_model(range(Sval[2, :Min], Sval[2, :Max], length = 200), rose_params))

pm_Co = DataFrame(Dd = range(Sval[3, :Min], Sval[3, :Max], length = 200),
                  JexPred = rose_model(range(Sval[3, :Min], Sval[3, :Max], length = 200), rose_params))

pm_Ni = DataFrame(Dd = range(Sval[4, :Min], Sval[4, :Max], length = 200),
                          JexPred = rose_model(range(Sval[4, :Min], Sval[4, :Max], length = 200), rose_params))


Mn_pressure = ExVals(1, "BCC", rose_params)
Fe_pressure = ExVals(2, "BCC", rose_params)
Co_pressure = ExVals(3, "FCC", rose_params)
Ni_pressure = ExVals(4, "FCC", rose_params)

##############################
# Plotting Layers
##############################
rose_layer = layer(x = Dig[:,:Dd], y = rose_model(Dig[:,:Dd], rose_params),
                       Geom.line,
                       Theme(default_color = "orange"))

roseFeCo_layer = layer(x = Dig[:, :Dd], y = rose_model(Dig[:, :Dd], roseFeCo_params),
                       Geom.line,
                       Theme(default_color = "teal"))

roseFerro_layer = layer(x = Dig[:, :Dd], y = rose_model(Dig[:, :Dd], roseFerro_params),
                      Geom.line,
                      Theme(default_color = "red"))

roseAll_layer = layer(x = Dig[:, :Dd], y = rose_model(Dig[:, :Dd], roseAll_params),
                       Geom.line,
                       Theme(default_color = "blue"))

poly2_layer = layer(x = Dig[:, :Dd], y = poly(Dig[:,:Dd], poly2_params),
                    Geom.line,
                    Theme(default_color = "teal"))

poly3_layer = layer(x = Dig[:, :Dd], y = poly(Dig[:,:Dd], poly3_params),
                    Geom.line,
                    Theme(default_color = "red"))

poly4_layer = layer(x = Dig[:, :Dd], y = poly(Dig[:,:Dd], poly4_params),
                    Geom.line,
                    Theme(default_color = "blue"))

BS_layer = layer(x = BS[:, :Dd], y = BS[:, :Ex],
                 label = BS[:, :Metal],
                 Geom.point, Geom.label)

Dig_layer = layer(x = Dig[:, :Dd], y = Dig[:, :Ex],
                  Geom.point)

NP_layer = layer(x = Dig[:, :Dd], y = nprval,
                 Geom.line,
                 Theme(default_color = "green"))

Chen_layer = layer(x = Chen[:, :Dd], y = rose_model(Chen[:, :Dd], rose_params),
                   Geom.point)

Mn_layer = layer(x = Mn_pressure[:, :Dd], y = Mn_pressure[:, :Ex],
                 Geom.line, order = 2,
                 Theme(line_width = 2pt,
                       default_color = "red"))

Fe_layer = layer(x = Fe_pressure[:, :Dd], y = Fe_pressure[:, :Ex],
                 Geom.line, order = 2,
                 Theme(line_width = 2pt,
                       default_color = "green"))

Co_layer = layer(x = Co_pressure[:, :Dd], y = Co_pressure[:, :Ex],
                 Geom.line, order = 2,
                 Theme(line_width = 2pt,
                       default_color = "blue"))

Ni_layer = layer(x = Ni_pressure[:, :Dd], y = Ni_pressure[:, :Ex],
                 Geom.line, order = 2,
                 Theme(line_width = 2pt,
                       default_color = "yellow"))


##############################
# Plotting Functions
##############################

polyplot = plot(poly3_layer, poly4_layer, poly2_layer, BS_layer,
     layer(xintercept = [0], Geom.vline),
     layer(yintercept = [0], Geom.hline),
     Guide.manual_color_key("Model",
                            ["Poly 2", "Poly 3", "Poly 4"],
                            ["teal", "red", "blue"]),
     Coord.cartesian(xmin = 1.2, xmax = 2.5, ymin = -1, ymax = 1),
     Theme(default_color = "black"),
     Guide.title("Polynomial Models for Digitized Data"),
     Guide.xlabel("D/d"),
     Guide.ylabel("Exchange Integral"))

roseplot = plot(rose_layer, NP_layer, BS_layer,
    layer(xintercept = [0], Geom.vline),
    layer(yintercept = [0], Geom.hline),
    Guide.manual_color_key("Model",
                           ["Rose", "Nonparametric"],
                           ["orange", "green"]),
    Coord.cartesian(xmin = 1.2, xmax = 2.5, ymin = -1, ymax = 1),
    Theme(default_color = "black"),
    Guide.title("Rose and Nonparametric Model for Digitized Data"),
    Guide.xlabel("D/d"),
    Guide.ylabel("Exchange Integral"))

fitplot = plot(BS_layer,
     rose4_layer,
     roseferro_layer,
     roseFeCo_layer,
     layer(xintercept = [0], Geom.vline),
     layer(yintercept = [0], Geom.hline),
     Guide.manual_color_key("Model",
                           ["All", "Ferromagnetic", "Fe, Co"],
                           ["red", "blue", "orange"]),
    Coord.cartesian(xmin = 1.2, xmax = 2.5, ymin = -1, ymax = 1),
    Theme(default_color = "black"),
    Guide.title("Rose Shift Model for 2,3,4 Data points"),
    Guide.xlabel("D/d"),
    Guide.ylabel("Exchange Integral"))

varpresplot = plot(rose_layer,
    layer(x = pm_Mn[:, :Dd], y = pm_Mn[:, :JexPred], Geom.line, Theme(default_color = "red"), order = 2),
    layer(x = pm_Fe[:, :Dd], y = pm_Fe[:, :JexPred], Geom.line, Theme(default_color = "blue"), order = 2),
    layer(x = pm_Co[:, :Dd], y = pm_Co[:, :JexPred], Geom.line, Theme(default_color = "green"), order = 2),
    layer(x = pm_Ni[:, :Dd], y = pm_Ni[:, :JexPred], Geom.line, Theme(default_color = "yellow"), order = 2),
    layer(x = BS[:, :Dd], y = BS[:, :Ex], label = BS[:, :Metal], Geom.point, Geom.label),
    layer(xintercept = [0], Geom.vline),
    layer(yintercept = [0], Geom.hline),
    Coord.cartesian(xmin = 1.2, xmax = 2.5, ymin = -1, ymax = 1),
    Theme(default_color = "black"),
    Guide.title("Varying Pressures - D/d +- 0.1"),
    Guide.xlabel("D/d"),
    Guide.ylabel("Exchange"))

rosepointplot = plot(rose_layer,
                      roseAll_layer,
                      roseFerro_layer,
                      roseFeCo_layer,
                      BS_layer,
                      layer(xintercept = [0], Geom.vline),
                      layer(yintercept = [0], Geom.hline),
                      Coord.cartesian(xmin = 1.2, xmax = 2.5, ymin = -1, ymax = 1),
                      Guide.manual_color_key("Model",
                                             ["Digitized Data", "Fe, Co", "Fe, Co, Ni", "All"],
                                             ["orange", "teal", "red", "blue"]),
                      Theme(default_color = "black"),
                      Guide.title("Rose Model Fits to 2,3,4 Data Points"),
                      Guide.xlabel("D/d"),
                      Guide.ylabel("Exchange"))

bulkpresplot = plot(rose_layer,
                    Fe_layer,
                    Co_layer,
                    Ni_layer,
                    BS_layer,
                    layer(xintercept = [0], Geom.vline),
                    layer(yintercept = [0], Geom.hline),
                    Coord.cartesian(xmin = 1.2, xmax = 2.5, ymin = -1, ymax = 1),
                    Theme(default_color = "black"),
                    Guide.title("Varying Pressures - 0.1 MPa to 10 GPa"),
                    Guide.xlabel("D/d"),
                    Guide.ylabel("Exchange"))


##############################
# Saving Data
##############################
rose_summary = DataFrame(xshift = Float64[], yshift = Float64[], mse = Float64[])
push!(rose_summary, (rose4_shift[1], rose4_shift[2], rose4_mse))
push!(rose_summary, (roseferro_shift[1], roseferro_shift[2], roseferro_mse))
push!(rose_summary, (roseFeCo_shift[1], roseFeCo_shift[2], roseFeCo_mse))

# println(rose_summary)
# print(rose_params)

# set_default_plot_size(6inch, 4inch)
# polyplot |> SVG("poly.svg")
# roseplot |> SVG("rose.svg")
# fitplot |> SVG("rosefits.svg")
# rosepointplot |> SVG("pointfits.svg")
# bulkpresplot |> SVG("varyingpressures.svg")
# CSV.write("shiftmse.csv", rose_summary)
# CSV.write("roseparams.csv", DataFrame(rose_params = rose_params))
# CSV.write("chen.csv", Chen)
