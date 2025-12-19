class EPBiLSTMCell(nn.Module):
    """
    EP BiLSTM cell with directional innovation terms, aligned to the paper notation.

    Paper variables and shapes

    x_t in R^D
    Forward states:  h_right_{t-1}, c_right_{t-1} in R^H
    Backward states: h_left_{t+1},  c_left_{t+1}  in R^H

    For each gate g in {f, i, o, g_tilde} and direction d in {right, left}:
      W_{x,d}^(g) in R^{D x H}
      W_{h,d}^(g) in R^{H x H}
      b_d^(g)     in R^H
      W_{e,d}^(g) in R^{H x H}

    Directional innovations
      e_t^right = x_t W_right - h_right_{t-1}
      e_t^left  = x_t W_left  - h_left_{t+1}

    Gate drives with innovation injection (per direction)
      preact_d^(g) = x_t W_{x,d}^(g) + h_d W_{h,d}^(g) + e_t^d W_{e,d}^(g) + b_d^(g)

    State updates remain standard LSTM in each direction.

    Implementation notes

    Row vector convention is used in code:
      x_t W is implemented as x_t @ W, with x_t shaped [B, D] and W shaped [D, H]

    Backward recurrence indexing:
      During the reversed loop at index t, the running variables (h_b_t, c_b_t) represent
      the carried backward states h_left_{t+1}, c_left_{t+1} before updating step t.
      After computing the gates and updating, they become h_left_t, c_left_t.

    Output dimension:
      The per step head maps concat([h_right_t, h_left_t]) to input_dim, so D_y equals D here.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim

        # Bias free directional input to hidden maps used to form innovations
        # e_t^right = W_right x_t - h_right_{t-1}
        # e_t^left  = W_left  x_t - h_left_{t+1}
        self.f_err_in2h = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_err_in2h = nn.Linear(input_dim, hidden_dim, bias=False)

        # Forward direction parameters
        self.f_forget_input, self.f_forget_hidden, self.f_forget_bias, self.f_forget_error = self.create_gate_parameters()
        self.f_input_input,  self.f_input_hidden,  self.f_input_bias,  self.f_input_error  = self.create_gate_parameters()
        self.f_output_input, self.f_output_hidden, self.f_output_bias, self.f_output_error = self.create_gate_parameters()
        self.f_cell_input,   self.f_cell_hidden,   self.f_cell_bias,   self.f_cell_error   = self.create_gate_parameters()

        # Backward direction parameters
        self.b_forget_input, self.b_forget_hidden, self.b_forget_bias, self.b_forget_error = self.create_gate_parameters()
        self.b_input_input,  self.b_input_hidden,  self.b_input_bias,  self.b_input_error  = self.create_gate_parameters()
        self.b_output_input, self.b_output_hidden, self.b_output_bias, self.b_output_error = self.create_gate_parameters()
        self.b_cell_input,   self.b_cell_hidden,   self.b_cell_bias,   self.b_cell_error   = self.create_gate_parameters()

        # Readout: y_hat_t = [h_right_t ; h_left_t] W_out + b_out
        # Here the head is implemented as Linear(2H, D)
        self.linear = nn.Linear(2 * hidden_dim, input_dim)

        # Init forget gate biases to +1 for both directions
        with torch.no_grad():
            self.f_forget_bias.fill_(1.0)
            self.b_forget_bias.fill_(1.0)

    def create_gate_parameters(self):
        W_x = nn.Parameter(torch.empty(self.input_dim,  self.hidden_dim))   # W_{x,d}^(g) in R^{D x H}
        W_h = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))   # W_{h,d}^(g) in R^{H x H}
        W_e = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))   # W_{e,d}^(g) in R^{H x H}
        nn.init.xavier_uniform_(W_x)
        nn.init.xavier_uniform_(W_h)
        nn.init.xavier_uniform_(W_e)
        b = nn.Parameter(torch.zeros(self.hidden_dim))                      # b_d^(g) in R^H
        return W_x, W_h, b, W_e

    def forward(self, x, h_f, c_f, h_b, c_b):
        """
        x   : [B, T, D]
        h_f : [B, H] initial forward hidden, corresponds to h_right_0 when T is processed from t=0
        c_f : [B, H] initial forward cell
        h_b : [B, H] initial backward hidden, corresponds to h_left_{T} carry when iterating from t=T-1 down to 0
        c_b : [B, H] initial backward cell
        """
        B, T, _ = x.shape

        # Forward chain, t from 0 to T minus 1
        f_h_list, f_c_list = [], []
        h_f_t, c_f_t = h_f, c_f
        for t in range(T):
            x_t = x[:, t]

            # Innovation e_t^right = W_right x_t - h_right_{t-1}
            e_right = self.f_err_in2h(x_t) - h_f_t

            f = torch.sigmoid(x_t @ self.f_forget_input + h_f_t @ self.f_forget_hidden + e_right @ self.f_forget_error + self.f_forget_bias)
            i = torch.sigmoid(x_t @ self.f_input_input  + h_f_t @ self.f_input_hidden  + e_right @ self.f_input_error  + self.f_input_bias)
            o = torch.sigmoid(x_t @ self.f_output_input + h_f_t @ self.f_output_hidden + e_right @ self.f_output_error + self.f_output_bias)
            g = torch.tanh   (x_t @ self.f_cell_input   + h_f_t @ self.f_cell_hidden   + e_right @ self.f_cell_error   + self.f_cell_bias)

            c_f_t = f * c_f_t + i * g
            h_f_t = o * torch.tanh(c_f_t)

            f_h_list.append(h_f_t.unsqueeze(1))
            f_c_list.append(c_f_t.unsqueeze(1))

        f_h_seq = torch.cat(f_h_list, dim=1)   # [B, T, H] forward h_right_t
        f_c_seq = torch.cat(f_c_list, dim=1)   # [B, T, H] forward c_right_t

        # Backward chain, t from T minus 1 down to 0
        # Before each update at index t, (h_b_t, c_b_t) represent (h_left_{t+1}, c_left_{t+1})
        # After update, they become (h_left_t, c_left_t)
        b_h_list_rev, b_c_list_rev = [], []
        h_b_t, c_b_t = h_b, c_b
        for t in reversed(range(T)):
            x_t = x[:, t]

            # Innovation e_t^left = W_left x_t - h_left_{t+1}
            e_left = self.b_err_in2h(x_t) - h_b_t

            f = torch.sigmoid(x_t @ self.b_forget_input + h_b_t @ self.b_forget_hidden + e_left @ self.b_forget_error + self.b_forget_bias)
            i = torch.sigmoid(x_t @ self.b_input_input  + h_b_t @ self.b_input_hidden  + e_left @ self.b_input_error  + self.b_input_bias)
            o = torch.sigmoid(x_t @ self.b_output_input + h_b_t @ self.b_output_hidden + e_left @ self.b_output_error + self.b_output_bias)
            g = torch.tanh   (x_t @ self.b_cell_input   + h_b_t @ self.b_cell_hidden   + e_left @ self.b_cell_error   + self.b_cell_bias)

            c_b_t = f * c_b_t + i * g
            h_b_t = o * torch.tanh(c_b_t)

            b_h_list_rev.append(h_b_t.unsqueeze(1))
            b_c_list_rev.append(c_b_t.unsqueeze(1))

        # Reverse collected backward outputs so index t aligns with x_t
        b_h_seq = torch.flip(torch.cat(b_h_list_rev, dim=1), dims=[1])  # [B, T, H] backward h_left_t
        b_c_seq = torch.flip(torch.cat(b_c_list_rev, dim=1), dims=[1])  # [B, T, H] backward c_left_t

        # Concatenate per time step: h_t = [h_right_t ; h_left_t], c_t = [c_right_t ; c_left_t]
        h_cat = torch.cat([f_h_seq, b_h_seq], dim=-1)  # [B, T, 2H]
        c_cat = torch.cat([f_c_seq, b_c_seq], dim=-1)  # [B, T, 2H]

        # Readout per time step, here output size equals input_dim
        out = self.linear(h_cat)  # [B, T, D]
        return out, h_cat, c_cat
