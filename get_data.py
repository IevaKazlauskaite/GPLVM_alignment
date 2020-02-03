import numpy as np

def get_data(N_seq, N_samples, D, noise_std=0.01, single_function=False,
             min_warp_coeff = 0.2, max_warp_coeff = 2.0):
    x = np.linspace(-1.0, 1.0, N_samples)

    if single_function:
        def func(x, n):
            return np.sinc(np.pi * x)
    else:
        def func(x, n):
            # return np.sinc(np.pi * x) if np.random.rand() < 0.5 else 0.6 * x ** 3
            if n < N_seq // 2:
                return np.sinc(np.pi * x)
            else:
                return 0.6 * x ** 3

    assert (N_seq > 1)

    x_warp = np.zeros([N_seq, N_samples])
    y_warp_noise_free = np.zeros([N_seq, N_samples, D])
    y_warp = np.zeros([N_seq, N_samples, D])

    x_warp[0] = x

    for n in range(0, N_seq):
        shifter = (max_warp_coeff - min_warp_coeff) * np.random.rand() + min_warp_coeff
        x_warp[n] = 2.0 * (np.linspace(0.0, 1.0, N_samples) ** shifter) - 1.0

        
    for n in range(N_seq):
        for d in range(D):
            y_warp_noise_free[n, :, d] = func(x_warp[n], n)
            y_warp[n, :, d] = y_warp_noise_free[n, :, d] + noise_std * np.random.randn(N_samples)

    return x_warp, y_warp, y_warp_noise_free
