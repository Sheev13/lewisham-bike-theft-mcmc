import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.spatial import distance_matrix
from scipy.stats import norm
from scipy.special import logsumexp, factorial
import matplotlib.cm as cm
import copy
from tqdm import tqdm
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd
from sklearn.model_selection import train_test_split


def GaussianKernel(x, l):
    """Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
    D = distance_matrix(x, x)
    return np.exp(-pow(D, 2) / (2 * pow(l, 2)))


def subsample(N, factor=4, seed=None, N_sub=None):
    assert not (factor is None and N_sub is None)
    if N_sub is None:
        assert factor >= 1, "Subsampling factor must be greater than or equal to one."
        N_sub = int(np.ceil(N / factor))
    assert isinstance(N_sub, int)
    if seed:
        np.random.seed(seed)
    idx = np.random.choice(
        N, size=N_sub, replace=False
    )  # Indexes of the randomly sampled points
    return idx


def get_G(N, idx):
    """Generate the observation matrix based on datapoint locations.
    Inputs:
        N - Length of vector to subsample
        idx - Indexes of subsampled coordinates
    Outputs:
        G - Observation matrix"""
    M = len(idx)
    G = np.zeros((M, N))
    for i in range(M):
        G[i, idx[i]] = 1
    return G


def probit(v):
    return np.where(v < 0, 0, 1)


def predict_t(samples):
    return norm.cdf(samples).mean(axis=0)  # Return p(t=1|samples)


def rmse(y, y_hat):
    return ((y - y_hat) ** 2).mean()


def hard_assign(probs):
    return np.where(probs < 0.5, 0, 1)


def mean_pred_error(t, p):
    return np.abs(t - p).mean()


def est_log_evidence(l, G, data, log_lik, n_samps=1000, D=16, coords=None):
    """Naive Monte Carlo estimator (arithmetic mean) for Marginal Likelihood"""
    if coords is None:
        coords = [(x, y) for y in np.linspace(0, 1, D) for x in np.linspace(0, 1, D)]
    N = G.shape[1]
    K = GaussianKernel(coords, l)
    z = np.random.randn(N, n_samps)
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    u = Kc @ z  # prior sample, shape (N, n_samps)
    ls = log_lik(u, data, G)  # shape (n_samps,)
    return logsumexp(ls) - np.log(n_samps)


def chib_est_log_evidence(u_star, Kc, data, G, post_samples, num_v_samps=None, beta=0.2):
    post_samples = post_samples.T
    num_u_samps = post_samples.shape[1]
    if num_v_samps is None:
        num_v_samps = num_u_samps
    N = Kc.shape[0]
    assert post_samples.shape[0] == N
    K = Kc @ Kc.T
    Kc_inv = np.linalg.inv(Kc)
    K_inv = Kc_inv.T @ Kc_inv

    # compute unnormalised posterior(x_star)
    unnorm_log_post = log_prior(u_star, K_inv) + log_probit_likelihood(u_star, data, G)

    # compute log of alpha(x_star, v_i) 's
    K_expanded_v = np.repeat(np.expand_dims(K, axis=0), num_v_samps, axis=0)
    # K_inv_expanded_v = np.repeat(np.expand_dims(K_inv, axis=0), num_v_samps, axis=0)
    u_star_for_v = np.repeat(np.expand_dims(u_star, axis=0), num_v_samps, axis=0)
    phi_dist_u_star_v = MultivariateNormal(
        torch.tensor(np.sqrt(1 - beta**2) * u_star_for_v),
        beta**2 * torch.tensor(K_expanded_v),
    )
    Vs = phi_dist_u_star_v.sample().T.numpy()
    # Vs = np.sqrt(1 - beta**2) * u_star_for_v + beta * Kc @ np.random.randn(
    #     N, num_v_samps
    # )
    phi_dist_v = MultivariateNormal(
        torch.tensor(np.sqrt(1 - beta**2) * Vs.T),
        beta**2 * torch.tensor(K_expanded_v),
    )
    lp_u_star_v = log_prior(u_star_for_v.T, K_inv)
    ll_u_star_v = log_probit_likelihood(u_star_for_v.T, data, G)
    lp_vs = log_prior(Vs, K_inv)
    ll_vs = log_probit_likelihood(Vs, data, G)
    lphi_star_v = phi_dist_v.log_prob(torch.tensor(u_star_for_v)).numpy()
    lphi_v_star = phi_dist_u_star_v.log_prob(torch.tensor(Vs.T)).numpy()
    log_alpha_v = np.minimum(ll_u_star_v - ll_vs, 0)
    # log_alpha_v = np.minimum(ll_u_star_v + lp_u_star_v - ll_vs - lp_vs, 0)
    # log_alpha_v = np.minimum(lphi_star_v - ll_u_star_v - lp_u_star_v - lphi_v_star + ll_vs + lp_vs, 0)
    
    

    # compute log of alpha(x_i, x_star) 's and of phi(x_star|x_i) 's
    K_expanded_u = np.repeat(np.expand_dims(K, axis=0), num_u_samps, axis=0)
    u_star_for_u = np.repeat(np.expand_dims(u_star, axis=0), num_u_samps, axis=0)
    lp_ui = log_prior(post_samples, K_inv)
    ll_ui = log_probit_likelihood(post_samples, data, G)
    lp_u_star_u = log_prior(u_star_for_u.T, K_inv)
    ll_u_star_u = log_probit_likelihood(u_star_for_u.T, data, G)
    phi_dist_ui = MultivariateNormal(
        torch.tensor(np.sqrt(1 - beta**2) * post_samples.T),
        beta**2 * torch.tensor(K_expanded_u),
    )
    lphi_star_ui = phi_dist_ui.log_prob(torch.tensor(u_star_for_u)).numpy()
    phi_dist_u_star_u = MultivariateNormal(
        torch.tensor(np.sqrt(1 - beta**2) * u_star_for_u),
        beta**2 * torch.tensor(K_expanded_u),
    )
    lphi_ui_star = phi_dist_u_star_u.log_prob(torch.tensor(post_samples.T)).numpy()
    
    log_alpha_u = np.minimum(ll_ui - ll_u_star_u, 0)
    # log_alpha_u = np.minimum(ll_ui + lp_ui - ll_u_star_u - lp_u_star_u, 0)
    # log_alpha_u = np.minimum(lphi_ui_star - ll_ui - lp_ui - lphi_star_ui + ll_u_star_u + lp_u_star_u, 0)

    log_numerator = unnorm_log_post + logsumexp(log_alpha_v) - np.log(num_v_samps)
    log_denominator = logsumexp(log_alpha_u + lphi_star_ui) - np.log(num_u_samps)
    
    return log_numerator - log_denominator
    
    # return unnorm_log_post, log_denominator - log_numerator + unnorm_log_post

    # return unnorm_log_post, logsumexp(log_alpha_v) - np.log(num_v_samps), logsumexp(log_alpha_u + lphi_star_ui) - np.log(num_u_samps)
    
    
def pred_dist_var(post_samps, lim=100):
    n = post_samps.shape[0]
    c = np.repeat(np.expand_dims(np.arange(lim), axis=-1), post_samps.shape[1], axis=-1)
    cs = np.repeat(np.expand_dims(c, axis=0), n, axis=0)
    u = np.expand_dims(post_samps, axis=1)
    ll = (-np.exp(u) + cs * u - np.log(factorial(cs)))
    pred_dist = np.exp(logsumexp(ll, axis=0) - np.log(n))
    E_c2 = (c**2 * pred_dist).sum(axis=0)
    Ec_2 = (c * pred_dist).sum(axis=0) ** 2
    
    return E_c2 - Ec_2
    


###--- Density functions ---###


def log_prior(u, K_inv):
    N_cont = np.log(2 * np.pi) * (u.shape[0] / 2)
    det_cont = -np.linalg.slogdet(K_inv)[1] / 2
    u_cont = (u.T @ K_inv @ u) / 2 
    if len(u_cont.shape) > 1:
        u_cont = u_cont[:,0]
    return -(N_cont + det_cont + u_cont)


def log_continuous_likelihood(u, v, G):
    M_cont = np.log(2 * np.pi) * (v.shape[0] / 2)
    diff = v - G @ u
    v_cont = diff.T @ diff / 2
    return -(M_cont + v_cont)


def log_probit_likelihood(u, t, G):
    phi = norm.cdf(G @ u)
    ids1 = np.where(t == 1)
    ids0 = np.where(t == 0)
    return np.log(phi[ids1]).sum(axis=0) + np.log(1 - phi[ids0]).sum(axis=0)


def log_poisson_likelihood(u, c, G):
    v = G @ u
    if len(v.shape) > 1:
        c = np.expand_dims(c, axis=1)
    return (-np.exp(v) + c * v - np.log(factorial(c))).sum(axis=0)


def log_continuous_target(u, y, K_inverse, G):
    return log_prior(u, K_inverse) + log_continuous_likelihood(u, y, G)


def log_probit_target(u, t, K_inverse, G):
    return log_prior(u, K_inverse) + log_probit_likelihood(u, t, G)


def log_poisson_target(u, c, K_inverse, G):
    return log_prior(u, K_inverse) + log_poisson_likelihood(u, c, G)


###--- MCMC ---###


def grw(log_target, u0, data, K, G, n_iters, beta, silence=False):
    """Gaussian random walk Metropolis-Hastings MCMC method
        for sampling from pdf defined by log_target.
    Inputs:
        log_target - log-target density
        u0 - initial sample
        data - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0

    # Inverse computed before the for loop for speed
    N = K.shape[0]
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    Kc_inverse = np.linalg.inv(Kc)
    K_inverse = Kc_inverse.T @ Kc_inverse

    lt_prev = log_target(u_prev, data, K_inverse, G)

    if not silence:
        print("GRW iterations:")
    for _ in tqdm(range(n_iters), disable=silence):

        u_new = u_prev + beta * Kc @ np.random.randn(
            N,
        )  # Propose new sample - use prior covariance, scaled by beta

        lt_new = log_target(u_new, data, K_inverse, G)

        log_alpha = min(
            lt_new - lt_prev, 0
        )  # Calculate acceptance probability based on lt_prev, lt_new
        log_u = np.log(np.random.random())

        # Accept/Reject
        if log_u <= log_alpha:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            lt_prev = lt_new
        else:
            X.append(u_prev)

    return np.array(X), acc / n_iters


def pcn(log_likelihood, u0, data, K, G, n_iters, beta, silence=False):
    """pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        data - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0

    N = K.shape[0]
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    ll_prev = log_likelihood(u_prev, data, G)

    if not silence:
        print("pCN iterations:")
    for _ in tqdm(range(n_iters), disable=silence):

        u_new = np.sqrt(1 - beta**2) * u_prev + beta * Kc @ np.random.randn(
            N,
        )  # Propose new sample using pCN proposal

        ll_new = log_likelihood(u_new, data, G)

        log_alpha = min(ll_new - ll_prev, 0)  # Calculate pCN acceptance probability
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = (
            log_u <= log_alpha
        )  # Compare log_alpha and log_u to accept/reject sample (accept should be boolean)
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            ll_prev = ll_new
        else:
            X.append(u_prev)

    return np.array(X), acc / n_iters


###--- Plotting ---###


def plot_3D(u, x, y, title=None, zlim=[-2.5, 2.5]):
    """Plot the latent variable field u given the list of x,y coordinates"""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(x, y, u, cmap="inferno", linewidth=0, antialiased=False)
    if title:
        plt.title(title)
    ax.set_zlim(zlim)
    ax.view_init(elev=25, azim=240)
    plt.show()


def plot_2D(counts, xi, yi, title=None, colors="inferno", classify=False, lim=None):
    """Visualise count data given the index lists"""
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under("k", alpha=0)
    fig, ax = plt.subplots()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if lim is None:
        lim = counts.max()
    im = ax.imshow(
        Z, origin="lower", cmap=my_cmap, clim=[min(0.0, counts.min()), lim]
    )
    if classify:
        class_0 = Patch(color=my_cmap(0.0), label=r"$t^*=0$")
        class_1 = Patch(color=my_cmap(1.0), label=r"$t^*=1$")
        ax.legend(handles=[class_0, class_1], bbox_to_anchor=(1.35, 0.7))
    else:
        fig.colorbar(im)
    if title:
        plt.title(title, fontsize=18)
    plt.show()


def plot_result(u, data, x, y, x_d, y_d, title=None, zlim=[-2.5, 2.5]):
    """Plot the latent variable field u with the observations,
    using the latent variable coordinate lists x,y and the
    data coordinate lists x_d, y_d"""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(x, y, u, cmap="inferno", linewidth=0, antialiased=False, alpha=0.6)
    ax.scatter(x_d, y_d, data, marker="x", color="darkgreen")
    if title:
        plt.title(title)
    ax.set_zlim(zlim)
    ax.view_init(elev=25, azim=240)
    plt.show()


###--- Data ---###


def generate_data(D=16, l=0.2, subsample_factor=4, N_sub=None):
    Dx, Dy = D, D
    N = Dx * Dy  # Total number of coordinates
    points = [
        (x, y) for y in np.arange(Dx) for x in np.arange(Dy)
    ]  # Indexes for the inference grid
    coords = [
        (x, y) for y in np.linspace(0, 1, Dy) for x in np.linspace(0, 1, Dx)
    ]  # Coordinates for the inference grid
    xi, yi = np.array([c[0] for c in points]), np.array(
        [c[1] for c in points]
    )  # Get x, y index lists
    x, y = np.array([c[0] for c in coords]), np.array(
        [c[1] for c in coords]
    )  # Get x, y coordinate lists

    ### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
    idx = subsample(N, factor=subsample_factor, N_sub=N_sub)
    M = len(idx)  # Total number of data points

    ### Generate K, the covariance of the Gaussian process, and sample from N(0,K) using a stable Cholesky decomposition
    K = GaussianKernel(coords, l)
    z = np.random.randn(
        N,
    )
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    u = Kc @ z

    ### Observation model: v = G(u) + e,   e~N(0,I)
    G = get_G(N, idx)
    v = G @ u + np.random.randn(M)

    return x, y, u, v, xi, yi, idx, coords, Kc, G


def get_spatial_data(subsample_factor=3, validate=False, val=0.3):
    # Read the data
    df = pd.read_csv('data.csv')

    # Generate the arrays needed from the dataframe
    data = np.array(df["bicycle.theft"])
    xi = np.array(df['xi'])
    yi = np.array(df['yi'])
    N = len(data)
    coords = [(xi[i],yi[i]) for i in range(N)]

    # Subsample the original data set
    idx = subsample(N, subsample_factor, seed=42)
    if validate:
        tr_idx, val_idx = train_test_split(idx, test_size=val, random_state=42)
        tr_G = get_G(N, tr_idx)
        tr_c = tr_G @ data
        val_G = get_G(N, val_idx)
        val_c = val_G @ data
        G = get_G(N,idx)
        c = G @ data
        return (tr_c, val_c, c), data, (tr_G, val_G, G), (tr_idx, val_idx, idx), coords, xi, yi
    else:
        G = get_G(N,idx)
        c = G @ data
        return c, data, G, idx, coords, xi, yi