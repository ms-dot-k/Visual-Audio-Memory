import torch
from torch import nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, radius=16.0, n_slot=88):
        super().__init__()

        self.key = nn.Parameter(torch.Tensor(n_slot, 512), requires_grad=True)
        nn.init.normal_(self.key, 0, 0.5)
        self.value = nn.Parameter(torch.Tensor(n_slot, 512), requires_grad=True)
        nn.init.normal_(self.value, 0, 0.5)

        self.q_embd = nn.Linear(512, 512)
        self.v_embd = nn.Linear(512, 512)

        self.fusion = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.5)

        self.radius = radius
        self.softmax = nn.Softmax(1)

    def forward(self, query, value=None, inference=False):
        # B, S, 512
        B, S, C = query.size()
        mer_query = query.view(B * S, -1)
        add_loss, tr_fusion, recon_loss = None, None, None

        key_norm = F.normalize(self.key, dim=1)
        embd_query = self.q_embd(mer_query)
        key_sim = F.linear(F.normalize(embd_query, dim=1), key_norm)
        key_add = self.softmax(self.radius * key_sim)

        vir_aud = torch.matmul(key_add, self.value.detach())

        te_fusion = torch.cat([query, vir_aud.view(B, S, -1)], 2)
        te_fusion = self.dropout(te_fusion)
        te_fusion = self.fusion(te_fusion)

        # Loss gen
        if not inference:
            mer_value = value.view(B * S, -1)
            embd_value = self.v_embd(mer_value.detach())
            value_norm = F.normalize(self.value, dim=1)
            value_sim = F.linear(F.normalize(embd_value, dim=1), value_norm)
            value_add = self.softmax(self.radius * value_sim)

            aud = torch.matmul(value_add, self.value)

            recon_loss = F.mse_loss(aud, mer_value.detach())
            recon_loss = recon_loss.unsqueeze(0)

            tr_fusion = torch.cat([query, value], 2)
            tr_fusion = self.dropout(tr_fusion)
            tr_fusion = self.fusion(tr_fusion)

            add_loss = F.kl_div(torch.log(key_add), value_add.detach(), reduction='batchmean')
            add_loss = add_loss.unsqueeze(0)

        return te_fusion, tr_fusion, recon_loss, add_loss
