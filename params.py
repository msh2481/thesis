some_params = {
    # Prior parameters
    # alpha ~ Beta(alpha_0, alpha_1)
    "alpha_0": 2.0,
    "alpha_1": 8.0,
    # k ~ Gamma(k_alpha, k_theta)
    "k_alpha": 3.0,
    "k_theta": 2.0,
    # sigma ~ Gamma(sigma_alpha, sigma_theta)
    "sigma_alpha": 1.0,
    "sigma_theta": 0.1,
    # wealth ~ Pareto(w_m, w_alpha)
    "w_m": 1000.0,
    "w_alpha": 1.5,
    # delay ~ Lognormal(mu_D, sigma_D)
    "mu_D": 0.0,
    "sigma_D": 0.5,
    # A ~ Logit-Normal(mu_A, sigma_A)
    "mu_A": -1.0,
    "sigma_A": 0.5,
    # delta ~ Gamma(delta_alpha, delta_theta)
    "delta_alpha": 2.0,
    "delta_theta": 0.01,
    # Process parameters
    "lambda": 5.0,
    "price_scale": 0.001,
    "use_levy": True,
    "levy_param": 1.5,
    "initial_price": 100.0,
}

eps = 1e-6
pure_insight_params = {
    "alpha_0": 10.0,
    "alpha_1": eps,
    "k_alpha": 1.0,
    "k_theta": 1.0,
    "sigma_alpha": eps,
    "sigma_theta": eps,
    "w_m": 1000.0,
    "w_alpha": 1.5,
    "mu_D": 0.0,
    "sigma_D": eps,
    "mu_A": 10.0,
    "sigma_A": eps,
    "delta_alpha": eps,
    "delta_theta": eps,
    "lambda": 50.0,
    "price_scale": 0.001,
    "use_levy": True,
    "levy_param": 1.4,
    "initial_price": 100.0,
}

some_insight_params = {
    "alpha_0": 1.0,
    "alpha_1": 1.0,
    "k_alpha": 1.0,
    "k_theta": 1000.0,
    "sigma_alpha": 1.0,
    "sigma_theta": 0.1,
    "w_m": 1000.0,
    "w_alpha": 1.5,
    "mu_D": 0.0,
    "sigma_D": 1.0,
    "mu_A": 0.0,
    "sigma_A": 1.0,
    "delta_alpha": 1.0,
    "delta_theta": 0.01,
    "lambda": 5.0,
    "price_scale": 0.001,
    "use_levy": True,
    "levy_param": 1.4,
    "initial_price": 100.0,
}
