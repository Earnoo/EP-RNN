import torch
import torch.nn as nn

class EPLSTMCell(nn.Module):
    """
    EP LSTM cell, matching the paper equations.

    Notation and conventions used here

    1. Shapes
       x_t in R^D, h_{t-1} and c_{t-1} in R^H
       W_x^(g) in R^{D x H}, W_h^(g) in R^{H x H}, b^(g) in R^H, W_e^(g) in R^{H x H}
       W in R^{D x H} used to form innovation e_t
       Readout W_out in R^{H x D_y}, b_out in R^{D_y}

    2. Row vector convention
       We implement x_t W_x^(g) as x_t @ W_x^(g), where x_t is treated as a row vector per batch item.

    3. Output dimension in this code
       The paper uses D_y as the output dimension.
       This implementation sets D_y = D by using nn.Linear(H, D) for the readout.
       This is consistent with the paper when your task predicts a D dimensional output at each step.
    """
    def __init__(self, D, H):
        super().__init__()
        self.D, self.H = D, H

        # Bias free input to hidden projection W in R^{D x H}
        # Innovation definition from the paper
        # e_t = x_t W - h_{t-1} in R^H
        self.W_in = nn.Linear(D, H, bias=False)

        # Gate parameters for g in {f, i, o, ĝ}
        # Each gate uses
        # W_x^(g) in R^{D x H}
        # W_h^(g) in R^{H x H}
        # b^(g) in R^H
        # W_e^(g) in R^{H x H}, innovation injection matrix
        self.Wx_f, self.Wh_f, self.b_f, self.We_f = self.create_gate_parameters()
        self.Wx_i, self.Wh_i, self.b_i, self.We_i = self.create_gate_parameters()
        self.Wx_o, self.Wh_o, self.b_o, self.We_o = self.create_gate_parameters()
        self.Wx_g, self.Wh_g, self.b_g, self.We_g = self.create_gate_parameters()

        # Readout layer, paper notation is
        # y_hat_t = h_t W_out + b_out, W_out in R^{H x D_y}, b_out in R^{D_y}
        # In this code, D_y is set equal to D, so W_out in R^{H x D} and b_out in R^D
        self.readout = nn.Linear(H, D)

        # Common initialization choice, set forget gate bias b^(f) to +1
        with torch.no_grad():
            self.b_f.fill_(1.0)

    def create_gate_parameters(self):
        # Explicit parameters, so we can implement the paper equations directly as x_t @ W_x^(g)
        W_x = nn.Parameter(torch.empty(self.D, self.H))     # W_x^(g) in R^{D x H}
        W_h = nn.Parameter(torch.empty(self.H, self.H))     # W_h^(g) in R^{H x H}
        W_e = nn.Parameter(torch.empty(self.H, self.H))     # W_e^(g) in R^{H x H}
        nn.init.xavier_uniform_(W_x)
        nn.init.xavier_uniform_(W_h)
        nn.init.xavier_uniform_(W_e)
        b = nn.Parameter(torch.zeros(self.H))               # b^(g) in R^H
        return W_x, W_h, b, W_e

    def forward(self, x_seq, h_prev, c_prev):
        # Inputs
        # x_seq has shape [B, T, D]
        # h_prev and c_prev have shape [B, H]
        B, T, _ = x_seq.shape
        h_seq, c_seq = [], []

        # Initialize recurrence with h_{t-1} = h_prev, c_{t-1} = c_prev
        h_t, c_t = h_prev, c_prev

        for t in range(T):
            # Current input x_t in R^D per batch item
            x_t = x_seq[:, t]

            # Innovation term from the paper
            # e_t = x_t W - h_{t-1} in R^H
            e_t = self.W_in(x_t) - h_t

            # Gate drives in the paper
            # preact_g = x_t W_x^(g) + h_{t-1} W_h^(g) + e_t W_e^(g) + b^(g)
            # Gate activations
            # f_t, i_t, o_t are sigmoid outputs in (0,1)^H
            # ĝ_t is tanh output in R^H
            f_t = torch.sigmoid(x_t @ self.Wx_f + h_t @ self.Wh_f + e_t @ self.We_f + self.b_f)
            i_t = torch.sigmoid(x_t @ self.Wx_i + h_t @ self.Wh_i + e_t @ self.We_i + self.b_i)
            o_t = torch.sigmoid(x_t @ self.Wx_o + h_t @ self.Wh_o + e_t @ self.We_o + self.b_o)
            g_t = torch.tanh   (x_t @ self.Wx_g + h_t @ self.Wh_g + e_t @ self.We_g + self.b_g)  # ĝ_t

            # State updates from the paper
            # c_t = f_t ⊙ c_{t-1} + i_t ⊙ ĝ_t
            # h_t = o_t ⊙ tanh(c_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            h_seq.append(h_t.unsqueeze(1))
            c_seq.append(c_t.unsqueeze(1))

        # Collect sequences
        h_seq = torch.cat(h_seq, dim=1)       # h_1..T, shape [B, T, H]
        c_seq = torch.cat(c_seq, dim=1)       # c_1..T, shape [B, T, H]

        # Readout, paper notation y_hat_t in R^{D_y}
        # Here D_y equals D, so output is [B, T, D]
        y_hat = self.readout(h_seq)
        return y_hat, h_seq, c_seq
