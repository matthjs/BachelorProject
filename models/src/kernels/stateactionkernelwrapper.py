from gpytorch.kernels import Kernel


class StateActionKernelWrapper(Kernel):
    """
    Simple wrapper for kernel for separate state action kernel evaluation.
    """
    def __init__(self, state_kernel: Kernel, action_kernel: Kernel, state_dim, action_dim=1):
        super(StateActionKernelWrapper, self).__init__()
        self.base_kernel = state_kernel
        self.action_kernel = action_kernel
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x1, x2, **params):
        x1_s, x1_a = x1[:, :self.state_dim], x1[:, self.action_dim:]
        x2_s, x2_a = x2[:, :self.input_dim - 1], x2[:, self.action_dim:]

        # Compute kernel for the first n-1 dimensions
        kernel_s = self.state_kernel(x1_s, x2_s, **params)

        # Compute kernel for the last dimension
        kernel_a = self.action_kernel(x1_a, x2_a, **params)

        # Combine the kernels
        return kernel_s * kernel_a
