import torch
import torch.nn as nn

class EPGRUCell(nn.Module):
    """
    EP GRU cell, matching the paper equations used by this implementation.

    Notation and conventions used here

    1. Shapes
       x_t in R^D, h_{t-1} in R^H
       For each g in {r, z, n}:
         W_x^(g) in R^{D x H}, W_h^(g) in R^{H x H}, b^(g) in R^H, W_e^(g) in R^{H x H}
       Innovation alignment W in R^{D x H}

    2. Row vector convention
       We implement x_t W_x^(g) as x_t @ W_x^(g), treating each batch row as a row vector.

    3. Candidate pathway convention in this code
       The reset gate is applied before the hidden and innovation linear maps:
         (r_t ⊙ h_{t-1}) W_h^(n) and (r_t ⊙ e_t) W_e^(n)
       This is the standard GRU form and matches the forward pass below.

    4. Output dimension in this code
       The paper uses D_y for the readout dimension.
       This implementation sets D_y = D by using nn.Linear(H, D) for per step output.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim

        # Bias free input to hidden alignment W in R^{D x H}
        # Innovation definition from the paper
        # e_t = x_t W - h_{t-1} in R^H
        self.err_in2h = nn.Linear(input_dim, hidden_dim, bias=False)

        # Gate parameters for g in {r, z, n}
        # Each gate uses W_x^(g), W_h^(g), b^(g), W_e^(g)
        self.reset_input,  self.reset_hidden,  self.reset_bias,  self.reset_error = self.create_gate_parameters()
        self.update_input, self.update_hidden, self.update_bias, self.update_error = self.create_gate_parameters()
        self.cand_input,   self.cand_hidden,   self.cand_bias,   self.cand_error  = self.create_gate_parameters()

        # Readout, paper notation is y_hat_t = h_t W_out + b_out with W_out in R^{H x D_y}
        # Here D_y equals D, so output is in R^D per time step
        self.linear = nn.Linear(hidden_dim, input_dim)

        # Optional init choice, biasing the update gate toward copying early on
        with torch.no_grad():
            self.update_bias.fill_(1.0)

    def create_gate_parameters(self):
        W_x = nn.Parameter(torch.empty(self.input_dim,  self.hidden_dim))   # W_x^(g) in R^{D x H}
        W_h = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))   # W_h^(g) in R^{H x H}
        W_e = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))   # W_e^(g) in R^{H x H}
        nn.init.xavier_uniform_(W_x)
        nn.init.xavier_uniform_(W_h)
        nn.init.xavier_uniform_(W_e)
        b = nn.Parameter(torch.zeros(self.hidden_dim))                      # b^(g) in R^H
        return W_x, W_h, b, W_e

    def forward(self, x, h):
        """
        x: [B, T, D]
        h: [B, H]  initial hidden state h_0
        Returns:
          out:   [B, T, D]   per step prediction y_hat_t (here D_y = D)
          h_seq: [B, T, H]
        """
        B, T, _ = x.shape
        h_seq = []

        for t in range(T):
            x_t = x[:, t]                     # x_t in R^D per batch item
            e_t = self.err_in2h(x_t) - h      # e_t = x_t W - h_{t-1} in R^H

            # Gates with innovation injection
            # r_t = sigma(x_t W_x^(r) + h_{t-1} W_h^(r) + e_t W_e^(r) + b^(r))
            # z_t = sigma(x_t W_x^(z) + h_{t-1} W_h^(z) + e_t W_e^(z) + b^(z))
            r = torch.sigmoid(x_t @ self.reset_input  + h @ self.reset_hidden  + e_t @ self.reset_error  + self.reset_bias)
            z = torch.sigmoid(x_t @ self.update_input + h @ self.update_hidden + e_t @ self.update_error + self.update_bias)

            # Candidate with reset applied before linear maps, matching this implementation
            # h_tilde = tanh(x_t W_x^(n) + (r_t ⊙ h_{t-1}) W_h^(n) + (r_t ⊙ e_t) W_e^(n) + b^(n))
            h_tilde = torch.tanh(
                x_t @ self.cand_input +
                (r * h)   @ self.cand_hidden +
                (r * e_t) @ self.cand_error +
                self.cand_bias
            )

            # Hidden update
            # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde
            h = (1 - z) * h + z * h_tilde

            h_seq.append(h.unsqueeze(1))

        h_seq = torch.cat(h_seq, dim=1)   # [B, T, H]
        out = self.linear(h_seq)          # [B, T, D]
        return out, h_seq
