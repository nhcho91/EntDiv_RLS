## Load packages and data
using LinearAlgebra, SparseArrays, Statistics, Plots, MAT, JLD2
cd(@__DIR__)

D = matread("cheetah3_leg_ID.mat")
# cheetah3_leg_ID.mat contains the copy of workspace obtained from running main.m in https://github.com/ROAM-Lab-ND/inertia_identification_minimal_examples
const B_stack = D["B_stack"]
const Bc_stack = D["Bc_stack"]
const Y_stack = D["Y_stack"]
const τ_stack = D["tau_stack"]
const Φ₀ = vec(D["pi_prior"])
const Φ_ref = vec(D["params"])
const b = vec(D["b"])
const bc = vec(D["bc"])
# q           = D["q"]
# qd          = D["qd"]
# qdd         = D["qdd"]
const dt_save = D["dt"]
const t = vec(D["t"])
# t = collect(t[1]:dt_save:t[end])

const dim_x = 9
const dim_τ = 3
const dim_t = length(t)

const θ̂₀ = [Φ₀; zeros(6)]
const θ_ref = [Φ_ref; b; bc]
const dim_θ = length(θ_ref)
const n_b = 6

const α = 0.1
const β = 1e-10
const γ = 1.0
const G = zeros(dim_θ, dim_θ)
const W = 1.0 / (dim_t - 101) / dim_τ * I(dim_τ)
const ϵ = 1e-20
const Ω₀ = zeros(dim_θ, dim_θ)
const P₀ = diagm([1.0 / α * ones(10 * n_b); 1.0 / β * ones(6)])
const Δ₀ = zeros(dim_θ)

Γ_stack = [Y_stack B_stack Bc_stack]
Γ_stack = [Γ_stack[(i - 1) * dim_τ + 1 : i * dim_τ, :] for i in eachindex(t)]
τ_stack = [τ_stack[(i - 1) * dim_τ + 1 : i * dim_τ] for i in eachindex(t)]

## Fuction definition
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

function R_grad(θ, θ₀, α, β, n_b)
    Φ = reshape(θ[1:10*n_b], 10, n_b)
    ψ = θ[10*n_b+1:end]
    Φ₀ = reshape(θ₀[1:10*n_b], 10, n_b)
    ψ₀ = θ₀[10*n_b+1:end]
    E = I(10)
    D_σ_grad = [[tr((ϕ2L(Φ₀[:, i]) \ ϕ2L(E[:, n])) - (ϕ2L(Φ[:, i]) \ ϕ2L(E[:, n]))) for n in 1:10] for i in 1:n_b]

    return [α * vcat(D_σ_grad...); β * (ψ - ψ₀)]
end

function R_Hess(θ, α, β, n_b)
    Φ = reshape(θ[1:10*n_b], 10, n_b)
    E = I(10)
    D_σ_Hess = [sparse([tr((ϕ2L(Φ[:, i]) \ ϕ2L(E[:, m])) * (ϕ2L(Φ[:, i]) \ ϕ2L(E[:, n]))) for m in 1:10, n in 1:10]) for i in 1:n_b]

    return blockdiag(α * D_σ_Hess..., sparse(β * I(6)))
end

function RLS_ldetdiv(θ̂_prev, Ω_prev, τ_current, Γ_current, W, G, ϵ, Δ₀, α, β, γ, θ̂₀, n_b)
    Ω_current = Ω_prev - G + Γ_current' * W * Γ_current

    Δ = Δ₀
    l = 0
    while true
        J_grad = Γ_current' * W * (Γ_current * θ̂_prev - τ_current) + Ω_current * Δ + R_grad(θ̂_prev + Δ, θ̂₀, α, β, n_b) - R_grad(θ̂_prev, θ̂₀, α, β, n_b)
        J_Hess = Ω_current + R_Hess(θ̂_prev + Δ, α, β, n_b)
        δ_current = -J_Hess \ J_grad
        λ² = -dot(J_grad, δ_current)
        if λ² / 2 <= ϵ
            break
        end
        Δ = Δ + γ * δ_current
        l = l + 1
    end

    θ̂_current = θ̂_prev + Δ

    return (θ̂_current, Ω_current, l)
end

function main(fₛ; flag_alg=1, flag_prog=0, flag_plot=0)
    # Downsampling
    dt_resample = 1 / fₛ
    id_t_cons = findall(x -> x >= t[1] + 0.05 && x < t[end] - 0.05, t)[1:round(Int, dt_resample / dt_save):end]
    t_cons = t[id_t_cons]
    dim_t_cons = length(id_t_cons)

    # Initialisation
    θ̂ = Vector{Vector{Float64}}(undef, dim_t_cons)   # zeros(dim_θ, dim_t_cons)
    τ̂ = similar(θ̂)   # zeros(dim_τ, dim_t_cons)
    l_f = zeros(dim_t_cons)

    θ̂_prev = θ̂₀
    Ω_prev = Ω₀
    P_prev = P₀

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
            (θ̂_current, Ω_current, l_current) = RLS_ldetdiv(θ̂_prev, Ω_prev, τ_current, Γ_current, W * dt_resample / dt_save, G, ϵ, Δ₀, α, β, γ, θ̂₀, n_b)
            l_f[k] = l_current
            Ω_prev = Ω_current
        else
            break
        end

        θ̂[k] = θ̂_current
        τ̂[k] = Γ_current * θ̂_current

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

    return (t_cons=t_cons, RMS_θ̃=RMS_θ̃, RMS_τ̃=RMS_τ̃, RMS_τ̃_f=RMS_τ̃_f, norm_θ̃=norm_θ̃, norm_τ̃=norm_τ̃, norm_τ̃_f=norm_τ̃_f, θ̂=θ̂, τ̂=τ̂, τ̂_f=τ̂_f, τ=τ, l_f=l_f)
end

## Single Run Simulation 
# sim_D = main(1e2; flag_alg=2, flag_prog=1, flag_plot=1)

## Benchmark Simulation
alg_list = [1, 2]
fₛ_list = [1, 10, 100, 1000]

sim_D = Array{NamedTuple}(undef, length(alg_list), length(fₛ_list))
for i_alg in eachindex(alg_list)
    for i_f in eachindex(fₛ_list)
        @show (i_alg, i_f)
        sim_D[i_alg, i_f] = main(fₛ_list[i_f]; flag_alg=alg_list[i_alg])
    end
end

jldsave("sim_D.jld2"; sim_D = sim_D)

##
id_t_valid = findall(x -> x >= t[1] + 0.05 && x < t[end] - 0.05, t)
t_valid = t[id_t_valid]
τ̂_BLS = D["tau_predict_entropic"][id_t_valid,:]

default(fontfamily="Computer Modern")

f_norm_θ̃ = plot(sim_D[1, 4].t_cons, [sim_D[1, 4].norm_θ̃ sim_D[2, 4].norm_θ̃], xlabel="\$t\$ [s]", ylabel="\$|| \\tilde{\\theta}\\left(t\\right) ||_{2}\$", ylims = (0, 10), label=["RLS-l2" "RLS-ldetdiv"], legend_position=:best)

f_norm_τ̃ = plot(sim_D[1, 4].t_cons, [sim_D[1, 4].norm_τ̃ sim_D[2, 4].norm_τ̃], xlabel="\$t\$ [s]", ylabel="\$|| \\tilde{\\tau}\\left(t;t\\right) ||_{2}\$ [Nm]", label=:false)

f_norm_τ̃_f = plot(sim_D[1, 4].t_cons, [sim_D[1, 4].norm_τ̃_f sim_D[2, 4].norm_τ̃_f], xlabel="\$t\$ [s]", ylabel="\$|| \\tilde{\\tau}\\left(t;t_{f}\\right) ||_{2}\$ [Nm]", label=:false)

f_l_f = plot(sim_D[2, 4].t_cons, sim_D[2, 4].l_f, xlabel="\$t\$ [s]", ylabel="\$l_{f}\$", label="RLS-ldetdiv", color=palette(:tab10)[2], legend_position=:best)

f_norm_l_f = plot(f_norm_θ̃, f_norm_τ̃, f_l_f, layout=(3, 1))
display(f_norm_l_f)
savefig(f_norm_l_f, "Fig_norm_l_f_1000Hz.pdf")

f_τ = plot(sim_D[1, 4].t_cons, hcat(sim_D[1, 4].τ̂...)', layout=(dim_τ, 1), label=[:false :false "RLS-l2"], xlabel="\$t\$ [s]", ylabel=["\$\\tau\\left(t;t\\right)_{\\textrm{Ab/Ad}}\$" "\$\\tau\\left(t;t\\right)_{\\textrm{Hip}}\$" "\$\\tau\\left(t;t\\right)_{\\textrm{Knee}}\$"])
plot!(f_τ, sim_D[2, 4].t_cons, hcat(sim_D[2, 4].τ̂...)', layout=(dim_τ, 1), label=[:false :false "RLS-ldetdiv"])
plot!(f_τ, t_valid, τ̂_BLS, layout=(dim_τ, 1), label=[:false :false "BLS-ldetdiv"])
plot!(f_τ, sim_D[1, 4].t_cons, hcat(sim_D[1, 4].τ...)', layout=(dim_τ, 1), color = palette(:tab10)[8], linewidth = 0.5, linealpha = 0.4, label=[:false :false "measured"])
display(f_τ)
savefig(f_τ, "Fig_tau_1000Hz.pdf")

f_τ_f = plot(sim_D[1, 4].t_cons, hcat(sim_D[1, 4].τ̂_f...)', layout=(dim_τ, 1), label=[:false :false "RLS-l2"], xlabel="\$t\$ [s]", ylabel=["\$\\tau\\left(t;t\\right)_{\\textrm{Ab/Ad}}\$ [Nm]" "\$\\tau\\left(t;t\\right)_{\\textrm{Hip}}\$ [Nm]" "\$\\tau\\left(t;t\\right)_{\\textrm{Knee}}\$ [Nm]"])
plot!(f_τ_f, sim_D[2, 4].t_cons, hcat(sim_D[2, 4].τ̂_f...)', layout=(dim_τ, 1), label=[:false :false "RLS-ldetdiv"])
plot!(f_τ_f, t_valid, τ̂_BLS, layout=(dim_τ, 1), label=[:false :false "BLS-ldetdiv"])
plot!(f_τ_f, sim_D[1, 4].t_cons, hcat(sim_D[1, 4].τ...)', layout=(dim_τ, 1), label=[:false :false "measured"])

f_RMS_θ̃ = plot(fₛ_list, [sim_D[i_alg, i_f].RMS_θ̃ for i_alg in eachindex(alg_list), i_f in eachindex(fₛ_list)]', xlabel="\$f_{s}\$ [Hz]", ylabel="\$|| \\tilde{\\theta}\\left(t\\right) ||_{rms}\$", xaxis=:log, label=:false)


flag_feas = zeros(length(alg_list), length(fₛ_list))
for i_alg in eachindex(alg_list)
    for i_f in eachindex(fₛ_list)
        Φ = reshape(sim_D[i_alg,i_f].θ̂[end][1:10*n_b], 10, n_b)
        flag_feas[i_alg, i_f] = prod([isposdef(ϕ2L(Φ[:, i])) for i in 1 : n_b])
    end
end

f_norm_θ̃_f = scatter(fₛ_list, [sim_D[i_alg, i_f].norm_θ̃[end] for i_alg in eachindex(alg_list), i_f in eachindex(fₛ_list)]', xlabel="\$f_{s}\$ [Hz]", ylabel="\$|| \\tilde{\\theta}\\left(t_{f}\\right) ||_{2}\$", xaxis=:log, xticks = fₛ_list, label=:false)

f_feas = scatter(fₛ_list, flag_feas', xlabel="\$f_{s}\$ [Hz]", ylabel="consistency", xaxis=:log, xticks = fₛ_list, label=:false)

f_RMS_τ̃ = scatter(fₛ_list, [sim_D[i_alg, i_f].RMS_τ̃ for i_alg in eachindex(alg_list), i_f in eachindex(fₛ_list)]', xlabel="\$f_{s}\$ [Hz]", ylabel="\$|| \\tilde{\\tau}\\left(t;t\\right) ||_{rms}\$ [Nm]", xaxis=:log, xticks = fₛ_list, label=:false)

f_RMS_τ̃_f = scatter(fₛ_list, [sim_D[i_alg, i_f].RMS_τ̃_f for i_alg in eachindex(alg_list), i_f in eachindex(fₛ_list)]', xlabel="\$f_{s}\$ [Hz]", ylabel="\$|| \\tilde{\\tau}\\left(t;t_{f}\\right) ||_{rms}\$ [Nm]", xaxis=:log, xticks = fₛ_list, label=["RLS-l2" "RLS-ldetdiv"])

f_RMS = plot(f_feas, f_norm_θ̃_f, f_RMS_τ̃, f_RMS_τ̃_f, layout=(4, 1), size = (600, 600))
display(f_RMS)
savefig(f_RMS, "Fig_feas_RMS.pdf")


