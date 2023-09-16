## Load packages and data
using LinearAlgebra, SparseArrays, Statistics, Plots, MAT, JLD2
cd(@__DIR__)

D = matread("cheetah3_leg_ID.mat")
# cheetah3_leg_ID.mat contains the copy of workspace obtained from running main.m in https://github.com/ROAM-Lab-ND/inertia_identification_minimal_examples
const B_stack = D["B_stack"]
const Bc_stack = D["Bc_stack"]
const Y_stack = D["Y_stack"]
τ_stack_D = D["tau_stack"]
const Φ₀ = vec(D["pi_prior"])
const Φ_ref = vec(D["params"])
const b = vec(D["b"])
const bc = vec(D["bc"])
const dt_save = D["dt"]
const t = vec(D["t"])

const dim_x = 9
const dim_τ = 3
const dim_t = length(t)

const θ̂₀ = [Φ₀; zeros(6)]
const θ_ref = [Φ_ref; b; bc]
const dim_θ = length(θ_ref)
const n_b = 6

const α = [5e1, 1e-1]
const β = 1e-3
const γ = 1.0
const G = zeros(dim_θ, dim_θ)
const W = 1.0 / (dim_t - 101) / dim_τ * I(dim_τ)
const ϵ = 1e-20
const Ω₀ = zeros(dim_θ, dim_θ)
const P₀ = diagm([ones(10 * n_b); 1.0 / β * ones(6)]) ./ α[1]
const Δ₀ = zeros(dim_θ)

Γ_stack_D = [Y_stack B_stack Bc_stack]
const Γ_stack = [Γ_stack_D[(i-1)*dim_τ+1:i*dim_τ, :] for i in eachindex(t)]
const τ_stack = [τ_stack_D[(i-1)*dim_τ+1:i*dim_τ] for i in eachindex(t)]

flag_RMS_pred = 1

# Fuction definition
function RLS_l2(θ̂_prev, P_prev, τ_current, Γ_current, W)
    K = P_prev * Γ_current' / (inv(W) + Γ_current * P_prev * Γ_current')
    θ̂_current = θ̂_prev + K * (τ_current - Γ_current * θ̂_prev)
    P_current = P_prev - K * Γ_current * P_prev

    return (θ̂_current, P_current)
end

function ϕ2L(ϕ)
    # ϕ = [m, h_x, h_y, h_z, I_xx, I_yy, I_zz, I_yz, I_xz, I_xy]
    m = ϕ[1]
    h = ϕ[2:4]
    I_rot = [ϕ[5] ϕ[10] ϕ[9]
        ϕ[10] ϕ[6] ϕ[8]
        ϕ[9] ϕ[8] ϕ[7]]

    return [0.5*tr(I_rot)*I(3)-I_rot h
        h' m]
end

function D_σ(θ₁, θ₂, n_b)
    Φ₁ = reshape(θ₁[1:10*n_b], 10, n_b)
    Φ₂ = reshape(θ₂[1:10*n_b], 10, n_b)
    return sum([-logdet(ϕ2L(Φ₁[:, i])) + logdet(ϕ2L(Φ₂[:, i])) + tr(ϕ2L(Φ₂[:, i]) \ ϕ2L(Φ₁[:, i])) - 4.0 for i in 1:n_b])
end

# function R_grad(θ, θ₀, α, β, n_b)
#     Φ = reshape(θ[1:10*n_b], 10, n_b)
#     ψ = θ[10*n_b+1:end]
#     Φ₀ = reshape(θ₀[1:10*n_b], 10, n_b)
#     ψ₀ = θ₀[10*n_b+1:end]
#     E = I(10)
#     D_σ_grad = [[tr((ϕ2L(Φ₀[:, i]) \ ϕ2L(E[:, n])) - (ϕ2L(Φ[:, i]) \ ϕ2L(E[:, n]))) for n in 1:10] for i in 1:n_b]

#     return [α * vcat(D_σ_grad...); β * (ψ - ψ₀)]
# end

# function R_Hess(θ, α, β, n_b)
#     Φ = reshape(θ[1:10*n_b], 10, n_b)
#     E = I(10)
#     D_σ_Hess = [sparse([tr((ϕ2L(Φ[:, i]) \ ϕ2L(E[:, m])) * (ϕ2L(Φ[:, i]) \ ϕ2L(E[:, n]))) for m in 1:10, n in 1:10]) for i in 1:n_b]

#     return blockdiag(α * D_σ_Hess..., sparse(β * I(6)))
# end

function R_grad_diff_Hess(θ_up, ψ_prev, L_prod_prev, L_unit, β, n_b)
    Φ_up = reshape(θ_up[1:10*n_b], 10, n_b)
    ψ_up = θ_up[10*n_b+1:end]

    L_prod_up = [ϕ2L(Φ_up[:, i]) \ L_unit[n] for i in 1:n_b, n in 1:10]

    D_σ_grad_diff = [[tr(L_prod_prev[i, n] - L_prod_up[i, n]) for n in 1:10] for i in 1:n_b]
    R_grad_diff = [vcat(D_σ_grad_diff...); β * (ψ_up - ψ_prev)]

    U_D_σ_Hess = [[m <= n ? tr(L_prod_up[i, m] * L_prod_up[i, n]) : 0.0 for m in 1:10, n in 1:10] for i in 1:n_b]
    D_σ_Hess = [sparse(U_D_σ_Hess[i] + U_D_σ_Hess[i]' - diagm(diag(U_D_σ_Hess[i]))) for i in 1:n_b]
    R_Hess = blockdiag(D_σ_Hess..., sparse(β * I(6)))

    return (R_grad_diff, R_Hess)
end

function RLS_ldetdiv(θ̂_prev, Ω_prev, τ_current, Γ_current, W, G, ϵ, Δ₀, α, β, γ, n_b, L_unit)
    Ω_current = Ω_prev - G + Γ_current' * W * Γ_current

    Δ = Δ₀
    l = 0
    Φ_prev = reshape(θ̂_prev[1:10*n_b], 10, n_b)
    ψ_prev = θ̂_prev[10*n_b+1:end]
    L_prod_prev = [ϕ2L(Φ_prev[:, i]) \ L_unit[n] for i in 1:n_b, n in 1:10]
    while true
        (R_grad_diff, R_Hess) = R_grad_diff_Hess(θ̂_prev + Δ, ψ_prev, L_prod_prev, L_unit, β, n_b)
        J_grad = Γ_current' * W * (Γ_current * θ̂_prev - τ_current) + Ω_current * Δ + α * R_grad_diff
        J_Hess = Ω_current + α * R_Hess

        δ_current = -J_Hess \ J_grad
        λ² = -dot(J_grad, δ_current)
        if λ² / 2.0 <= ϵ
            break
        end
        Δ = Δ + γ * δ_current
        l = l + 1
    end

    θ̂_current = θ̂_prev + Δ

    return (θ̂_current, Ω_current, l)
end

function main(fₛ; flag_alg=1, flag_prog=0, flag_plot=0, flag_RMS_pred=0)
    # Downsampling
    dt_resample = 1 / fₛ
    id_t_cons = findall(x -> x >= t[1] + 0.05 && x < t[end] - 0.05, t)[1:round(Int, dt_resample / dt_save):end]
    t_cons = t[id_t_cons]
    dim_t_cons = length(id_t_cons)

    # Initialisation
    θ̂ = Vector{Vector{Float64}}(undef, dim_t_cons)   # zeros(dim_θ, dim_t_cons)
    τ̂ = similar(θ̂)   # zeros(dim_τ, dim_t_cons)
    l_f = zeros(dim_t_cons)
    T_elap = similar(l_f)
    RMS_τ̃_pred = similar(l_f)

    θ̂_prev = θ̂₀
    Ω_prev = Ω₀
    P_prev = P₀

    L_unit = [ϕ2L(I(10)[:, n]) for n in 1:10]

    # Loop
    for k in 1:dim_t_cons
        if flag_prog == 1
            println("Training progress = $(k) / $(dim_t_cons)\n")
        end

        τ_current = τ_stack[id_t_cons[k]]
        Γ_current = Γ_stack[id_t_cons[k]]

        if flag_alg == 1
            (θ̂_current, P_current) = RLS_l2(θ̂_prev, P_prev, τ_current, Γ_current, W * dt_resample / dt_save)
            P_prev = P_current

        elseif flag_alg == 2
            v_current = @timed RLS_ldetdiv(θ̂_prev, Ω_prev, τ_current, Γ_current, W * dt_resample / dt_save, G, ϵ, Δ₀, α[flag_alg], β, γ, n_b, L_unit)
            (θ̂_current, Ω_current, l_current) = v_current.value
            l_f[k] = l_current
            T_elap[k] = v_current.time - v_current.gctime
            Ω_prev = Ω_current
        else
            break
        end

        θ̂[k] = θ̂_current
        τ̂[k] = Γ_current * θ̂_current
        if flag_RMS_pred == 1
            τ̃_pred = [Γ_stack[id_t_cons[j]] * θ̂_current - τ_stack[id_t_cons[j]] for j in k:dim_t_cons]
            RMS_τ̃_pred[k] = sqrt(mean(dot.(τ̃_pred, τ̃_pred)))
        end

        θ̂_prev = θ̂_current
    end

    # Performance Metrics
    θ̃ = θ̂ - [θ_ref for k in 1:dim_t_cons]
    τ = [τ_stack[id_t_cons[k]] for k in 1:dim_t_cons]
    τ̃ = τ̂ - τ
    τ̂_f = [Γ_stack[id_t_cons[k]] * θ̂[end] for k in 1:dim_t_cons]
    τ̃_f = τ̂_f - τ

    RMS_θ̃ = sqrt(mean(dot.(θ̃, θ̃)))
    RMS_τ̃ = sqrt(mean(dot.(τ̃, τ̃)))
    RMS_τ̃_f = sqrt(mean(dot.(τ̃_f, τ̃_f)))

    norm_θ̃ = norm.(θ̃)
    norm_τ̃ = norm.(τ̃)
    norm_τ̃_f = norm.(τ̃_f)

    D_Φ = [flag_alg == 2 ? D_σ(θ̂[k], θ_ref, n_b) : NaN for k in 1:dim_t_cons]
    D_ψ = [norm( (θ̂[k]-θ_ref)[10*n_b+1:end] ./ abs.(θ_ref[10*n_b+1:end]) ) for k in 1:dim_t_cons]

    # Save and Plot    
    if flag_plot == 1
        default(fontfamily="Computer Modern")
        alg_string = ["RLS-l2", "RLS-ldetdiv"]

        f_norm = plot(t_cons, [norm_θ̃ norm_τ̃], layout=(2, 1), xlabel="\$t\$ [s]", ylabel=["\$|| \\tilde{\\theta} ||_{2}\$" "\$|| \\tilde{\\tau} ||_{2}\$ [Nm]"], label=alg_string[flag_alg], legend_position=:best)
        display(f_norm)
        savefig(f_norm, "Fig_norm_$(flag_alg).pdf")

        f_τ = plot(t_cons, hcat(τ...)', layout=(dim_τ, 1), label=[:false :false "measured"])
        plot!(f_τ, t_cons, hcat(τ̂...)', layout=(dim_τ, 1), xlabel="\$t\$ [s]", ylabel=["Ab/Ad [Nm]" "Hip [Nm]" "Knee [Nm]"], label=[:false :false alg_string[flag_alg]], legend_position=:bottomleft)
        display(f_τ)
        savefig(f_τ, "Fig_tau_$(flag_alg).pdf")

        f_τ_f = plot(t_cons, hcat(τ...)', layout=(dim_τ, 1), label=[:false :false "measured"])
        plot!(f_τ_f, t_cons, hcat(τ̂_f...)', layout=(dim_τ, 1), xlabel="\$t\$ [s]", ylabel=["Ab/Ad [Nm]" "Hip [Nm]" "Knee [Nm]"], label=[:false :false alg_string[flag_alg] * "-final"], legend_position=:bottomleft)
        display(f_τ_f)
        savefig(f_τ_f, "Fig_tau_f_$(flag_alg).pdf")
    end

    return (t_cons=t_cons, RMS_θ̃=RMS_θ̃, RMS_τ̃=RMS_τ̃, RMS_τ̃_f=RMS_τ̃_f, norm_θ̃=norm_θ̃, norm_τ̃=norm_τ̃, norm_τ̃_f=norm_τ̃_f, θ̂=θ̂, τ̂=τ̂, τ̂_f=τ̂_f, τ=τ, l_f=l_f, T_elap=T_elap, RMS_τ̃_pred=RMS_τ̃_pred, D_Φ=D_Φ, D_ψ=D_ψ)
end

## Single Run 
# sim_D = main(1e3; flag_alg=2, flag_prog=0, flag_plot=1)

## Benchmark
alg_list = [1, 2]
fₛ_list = [1, 10, 100, 1000]

sim_D = Array{NamedTuple}(undef, length(alg_list), length(fₛ_list))
for i_alg in eachindex(alg_list)
    for i_f in eachindex(fₛ_list)
        @show (i_alg, i_f)
        sim_D[i_alg, i_f] = main(fₛ_list[i_f]; flag_alg=alg_list[i_alg], flag_prog=1, flag_RMS_pred=flag_RMS_pred)
    end
end

jldsave("sim_D.jld2"; sim_D=sim_D)

## Plot Results
sim_D = load("sim_D.jld2", "sim_D")

id_t_valid = findall(x -> x >= t[1] + 0.05 && x < t[end] - 0.05, t)
t_valid = t[id_t_valid]
dim_t_valid = length(id_t_valid)
τ̂_BLS = D["tau_predict_entropic"][id_t_valid, :]

default(fontfamily="Computer Modern")

f_norm_θ̃ = plot(sim_D[1, 4].t_cons, [sim_D[1, 4].norm_θ̃ sim_D[2, 4].norm_θ̃], xlabel="\$t\$ [s]", ylabel="\$|| \\tilde{\\theta}\\left(t\\right) ||_{2}\$", label=:false, legend_position=:best)

f_norm_τ̃ = plot(sim_D[1, 4].t_cons, [sim_D[1, 4].norm_τ̃ sim_D[2, 4].norm_τ̃], xlabel="\$t\$ [s]", ylabel="\$|| \\tilde{\\tau}\\left(t;t\\right) ||_{2}\$ [Nm]", label=:false)

f_norm_τ̃_f = plot(sim_D[1, 4].t_cons, [sim_D[1, 4].norm_τ̃_f sim_D[2, 4].norm_τ̃_f], xlabel="\$t\$ [s]", ylabel="\$|| \\tilde{\\tau}\\left(t;t_{f}\\right) ||_{2}\$ [Nm]", label=:false)

f_l_f = plot(sim_D[2, 4].t_cons, sim_D[2, 4].l_f, xlabel="\$t\$ [s]", ylabel="\$l_{f}\$", label="RLS-ldetdiv", color=palette(:tab10)[2], legend_position=:best)

f_T = plot(sim_D[2, 4].t_cons, sim_D[2, 4].T_elap, xlabel="\$t\$ [s]", ylabel="\$T_{comp}\$", label="RLS-ldetdiv", color=palette(:tab10)[2], legend_position=:best)

f_D_Φ = plot(sim_D[1, 4].t_cons, [sim_D[1, 4].D_Φ sim_D[2, 4].D_Φ sim_D[2, 4].D_Φ[1]*ones(length(sim_D[1,4].t_cons))], xlabel="\$t\$ [s]", ylabel="\$D_{\\Phi}\$", label=["RLS-l2" "RLS-ldetdiv" "prior"], legend_position=:topright)

f_D_ψ = plot(sim_D[1, 4].t_cons, [sim_D[1, 4].D_ψ sim_D[2, 4].D_ψ], xlabel="\$t\$ [s]", ylabel="\$D_{\\psi}\$", label=:false, legend_position=:best)

# f_D_Φψ = plot(f_D_Φ, f_D_ψ, layout=(2, 1))
# display(f_D_Φψ)
# savefig(f_D_Φψ, "Fig_D_Phipsi_1000Hz.pdf")

# f_norm_l_f = plot(f_norm_θ̃, f_norm_τ̃, f_l_f, layout=(3, 1))
# display(f_norm_l_f)
# savefig(f_norm_l_f, "Fig_norm_l_f_1000Hz.pdf")

# f_norm = plot(f_norm_θ̃, f_norm_τ̃, layout=(2, 1))
# display(f_norm)
# savefig(f_norm, "Fig_norm_1000Hz.pdf")

# f_norm_T = plot(f_norm_θ̃, f_norm_τ̃, f_T, layout=(3, 1))
# display(f_norm_T)
# savefig(f_norm_T, "Fig_norm_T_1000Hz.pdf")


if flag_RMS_pred == 1
    RMS_τ̃_pred_prior = zeros(dim_t_valid)
    for k in 1:dim_t_valid
        τ̃_pred_prior = [Γ_stack[id_t_valid[j]] * θ̂₀ - τ_stack[id_t_valid[j]] for j in k:dim_t_valid]
        RMS_τ̃_pred_prior[k] = sqrt(mean(dot.(τ̃_pred_prior, τ̃_pred_prior)))
    end

    f_RMS_τ̃_pred = plot(sim_D[1, 4].t_cons, [sim_D[1, 4].RMS_τ̃_pred sim_D[2, 4].RMS_τ̃_pred RMS_τ̃_pred_prior], xlabel="\$t\$ [s]", ylabel="\$|| \\tilde{\\tau}\\left(t\\prime;t\\right) ||_{rms}^{\\left[t,t_{f}\\right]}\$", label=:false)

    f_D_Φψ_RMS_τ̃_pred = plot(f_D_Φ, f_D_ψ, f_norm_θ̃, f_RMS_τ̃_pred, layout=(4, 1), size=(600,500))
    display(f_D_Φψ_RMS_τ̃_pred)
    savefig(f_D_Φψ_RMS_τ̃_pred, "Fig_D_Phipsi_RMS_tau_pred_1000Hz.pdf")
end


f_τ = plot(sim_D[1, 4].t_cons, hcat(sim_D[1, 4].τ̂...)', layout=(dim_τ, 1), label=[:false :false "RLS-l2"], xlabel="\$t\$ [s]", ylabel=["\$\\tau\\left(t;t\\right)_{\\textrm{Ab/Ad}}\$" "\$\\tau\\left(t;t\\right)_{\\textrm{Hip}}\$" "\$\\tau\\left(t;t\\right)_{\\textrm{Knee}}\$"])
plot!(f_τ, sim_D[2, 4].t_cons, hcat(sim_D[2, 4].τ̂...)', layout=(dim_τ, 1), label=[:false :false "RLS-ldetdiv"])
plot!(f_τ, t_valid, τ̂_BLS, layout=(dim_τ, 1), label=[:false :false "BLS-ldetdiv"])
plot!(f_τ, sim_D[1, 4].t_cons, hcat(sim_D[1, 4].τ...)', layout=(dim_τ, 1), color=palette(:tab10)[8], linewidth=0.5, linealpha=0.4, label=[:false :false "measured"])
display(f_τ)
savefig(f_τ, "Fig_tau_1000Hz.pdf")

# f_τ_f = plot(sim_D[1, 4].t_cons, hcat(sim_D[1, 4].τ̂_f...)', layout=(dim_τ, 1), label=[:false :false "RLS-l2"], xlabel="\$t\$ [s]", ylabel=["\$\\tau\\left(t;t\\right)_{\\textrm{Ab/Ad}}\$ [Nm]" "\$\\tau\\left(t;t\\right)_{\\textrm{Hip}}\$ [Nm]" "\$\\tau\\left(t;t\\right)_{\\textrm{Knee}}\$ [Nm]"])
# plot!(f_τ_f, sim_D[2, 4].t_cons, hcat(sim_D[2, 4].τ̂_f...)', layout=(dim_τ, 1), label=[:false :false "RLS-ldetdiv"])
# plot!(f_τ_f, t_valid, τ̂_BLS, layout=(dim_τ, 1), label=[:false :false "BLS-ldetdiv"])
# plot!(f_τ_f, sim_D[1, 4].t_cons, hcat(sim_D[1, 4].τ...)', layout=(dim_τ, 1), label=[:false :false "measured"])

# f_RMS_θ̃ = plot(fₛ_list, [sim_D[i_alg, i_f].RMS_θ̃ for i_alg in eachindex(alg_list), i_f in eachindex(fₛ_list)]', xlabel="\$f_{s}\$ [Hz]", ylabel="\$|| \\tilde{\\theta}\\left(t\\right) ||_{rms}\$", xaxis=:log, label=:false)

flag_feas = zeros(length(alg_list), length(fₛ_list))
for i_alg in eachindex(alg_list)
    for i_f in eachindex(fₛ_list)
        Φ = reshape(sim_D[i_alg, i_f].θ̂[end][1:10*n_b], 10, n_b)
        flag_feas[i_alg, i_f] = prod([isposdef(ϕ2L(Φ[:, i])) for i in 1:n_b])
    end
end

# f_norm_θ̃_f = scatter(fₛ_list, [sim_D[i_alg, i_f].norm_θ̃[end] for i_alg in eachindex(alg_list), i_f in eachindex(fₛ_list)]', xlabel="\$f_{s}\$ [Hz]", ylabel="\$|| \\tilde{\\theta}\\left(t_{f}\\right) ||_{2}\$", xaxis=:log, xticks=fₛ_list, label=:false)

# f_feas = scatter(fₛ_list, flag_feas', xlabel="\$f_{s}\$ [Hz]", ylabel="\$\\mathcal{I}_{cons}\$", xaxis=:log, ylims=(-0.1, 1.1), xticks=fₛ_list, yticks=[0, 1], label=:false)

# f_RMS_τ̃ = scatter(fₛ_list, [sim_D[i_alg, i_f].RMS_τ̃ for i_alg in eachindex(alg_list), i_f in eachindex(fₛ_list)]', xlabel="\$f_{s}\$ [Hz]", ylabel="\$|| \\tilde{\\tau}\\left(t;t\\right) ||_{rms}^{\\left[t_{0},t_{f}\\right]}\$", xaxis=:log, xticks=fₛ_list, label=:false)

# f_RMS_τ̃_f = scatter(fₛ_list, [sim_D[i_alg, i_f].RMS_τ̃_f for i_alg in eachindex(alg_list), i_f in eachindex(fₛ_list)]', xlabel="\$f_{s}\$ [Hz]", ylabel="\$|| \\tilde{\\tau}\\left(t;t_{f}\\right) ||_{rms}^{\\left[t_{0},t_{f}\\right]}\$", xaxis=:log, xticks=fₛ_list, label=["RLS-l2" "RLS-ldetdiv"])

# f_RMS = plot(f_feas, f_norm_θ̃_f, f_RMS_τ̃, f_RMS_τ̃_f, layout=(4, 1), size=(600, 400))
# display(f_RMS)
# savefig(f_RMS, "Fig_feas_RMS.pdf")


# computation time statistics
id_N1 = findall(x -> x == 1, sim_D[2, 4].l_f)
id_N2 = findall(x -> x == 2, sim_D[2, 4].l_f)
T_N1 = sim_D[2, 4].T_elap[id_N1]
T_N2 = sim_D[2, 4].T_elap[id_N2]
@show (mean(T_N1), std(T_N1), mean(T_N2), std(T_N2));

## write MAT files for joint space inertia matrix positive definiteness test using Prof Wensing's original codebase (HandC.m)
θ̂_f_RLS_l2 = hcat([sim_D[1, i].θ̂[end] for i in eachindex(fₛ_list)]...)
θ̂_f_RLS_ldetdiv = hcat([sim_D[2, i].θ̂[end] for i in eachindex(fₛ_list)]...)
θ̂_RLS_l2 = [sim_D[1, i].θ̂ for i in eachindex(fₛ_list)]
θ̂_RLS_ldetdiv = [sim_D[2, i].θ̂ for i in eachindex(fₛ_list)]

file = matopen("./inertia_identification_minimal_examples/data/theta_hat.mat", "w")
write(file, "theta_hat_final_RLS_l2", θ̂_f_RLS_l2)
write(file, "theta_hat_final_RLS_ldetdiv", θ̂_f_RLS_ldetdiv)
write(file, "theta_hat_RLS_l2", θ̂_RLS_l2)
write(file, "theta_hat_RLS_ldetdiv", θ̂_RLS_ldetdiv)
close(file)

# Go to MATLAB and run ./inertia_identification_minimal_examples/Joint_I_PD_test.m (attached below)
#= 
% Need to run this file with its enclosing folder as the working directory
fileInfo = dir(matlab.desktop.editor.getActiveFilename);
cd(fileInfo.folder);

clc
clear
close all
init_path

%% Load Data
load('CheetahSysID.mat');
model = Cheetah3LegModel();

%% Joint Space Inertia Matrix Positive Definiteness Check -- w.r.t. running estimate
load("theta_hat.mat") % <---- Obtained from Julia code

test_summary = zeros(2, size(theta_hat_final_RLS_l2, 2));
for i_alg = 1 : 2
    for i_f = 1 : size(theta_hat_final_RLS_l2, 2)
        if i_alg == 1
            theta = theta_hat_RLS_l2{i_f};
        elseif i_alg == 2
            theta = theta_hat_RLS_ldetdiv{i_f};
        end

        dt_resample = 1 / 10^(i_f-1);
        dt_save = 1e-3;
        t = 0:dt_save:28;
        id_t_valid = find(t>=0.05 & t<28-0.05); 
        id_t_cons = id_t_valid(1:dt_resample/dt_save:end);
        q_cons = q(id_t_cons);
        qd_cons = qd(id_t_cons);

        M = cell(length(q_cons),1);
        TF_M_pd = zeros(length(q_cons),1);
        for i = 1 : length(q_cons)
            Phi = reshape(theta{i}(1:end-6), 10, []);
            for j = 1 : model.NB
                model.I{j} = inertiaVecToMat(Phi(:,j));
                model.I_rotor{j} = inertiaVecToMat(Phi(:,j+3));
            end

            [H,C,info] = HandC( model, q_cons{i}, qd_cons{i} );
            M{i} = H;

            TF_M_pd(i) = issymmetric(H) && isreal(eig(H)) && prod(eig(H) > 0);
        end
        figure; plot(TF_M_pd), title(sprintf("i_alg = %d, i_f = %d", i_alg, i_f))
        test_summary(i_alg, i_f) = prod(TF_M_pd);
    end
end

%% Joint Space Inertia Matrix Positive Definiteness Check -- w.r.t. final estimate
load("theta_hat.mat") % <---- Obtained from Julia code

test_summary = zeros(2, size(theta_hat_final_RLS_l2, 2));
for i_alg = 1 : 2
    if i_alg == 1
        Phi = theta_hat_final_RLS_l2(1:60,:);
    elseif i_alg == 2
        Phi = theta_hat_final_RLS_ldetdiv(1:60,:);
    end

    for i_f = 1 : size(theta_hat_final_RLS_l2, 2)
        Phi_f = reshape(Phi(:, i_f), 10, []);
        for i = 1 : model.NB
            model.I{i} = inertiaVecToMat(Phi_f(:,i));
            model.I_rotor{i} = inertiaVecToMat(Phi_f(:,i+3));
        end

        M = cell(length(q),1);
        TF_M_pd = zeros(length(q),1);
        for i = 1 : length(q)
            [H,C,info] = HandC( model, q{i}, qd{i} );
            M{i} = H;

            TF_M_pd(i) = issymmetric(H) & isreal(eig(H)) & prod(eig(H) > 0);
        end
        figure; plot(TF_M_pd), title(sprintf("i_alg = %d, i_f = %d", i_alg, i_f))
        test_summary(i_alg, i_f) = prod(TF_M_pd);
    end
end
=#

