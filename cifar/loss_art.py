import torch
import torch.nn as nn

ranking_loss = nn.SoftMarginLoss()
cos = nn.CosineSimilarity(dim=1, eps=1e-15)

def hard_example_mining(dist_mat_p , dist_mat_n , labels, return_inds=False):
    N = dist_mat_p.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    
    dist_ap, relative_p_inds = torch.max(
        dist_mat_p[is_pos].contiguous().view(N, -1), 1, keepdim=True)
   
    dist_an, relative_n_inds = torch.min(
        dist_mat_n[is_pos].contiguous().view(N, -1), 1, keepdim=True)
 
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an

def cosine_dist(x, g1, g2):
    dot_p = x @ g1.t()
    norm1 = torch.norm(x, 2, 1) + 1e-8
    norm2 = torch.norm(g1, 2, 1)+ 1e-8
    dot_p = torch.div(dot_p, norm1.unsqueeze(1))
    dot_p = torch.div(dot_p, norm2)
    
    dot_n = x @ g2.t()
    norm3 = torch.norm(g2, 2, 1)+ 1e-8
    dot_n = torch.div(dot_n, norm1.unsqueeze(1))
    dot_n = torch.div(dot_n, norm3)
    
    return 1.0 - dot_p , 1.0 - dot_n
 
def exemplar_loss_fn(x , g1, g2 , y, a) :

    dist_mat_p , dist_mat_n = cosine_dist(x, g1, g2)
    dist_ap, dist_an = hard_example_mining(dist_mat_p , dist_mat_n, a, return_inds=False)
    y = dist_an.new().resize_as_(dist_an).fill_(1)
    loss = ranking_loss(dist_an - dist_ap, y)
    return loss