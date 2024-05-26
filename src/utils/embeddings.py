import torch

# Learned embeddings
class Cube(torch.nn.Module):
    def __init__(self, res=40, feature_dim=24, half_size=2):
        super(Cube, self).__init__()
        self.dim = feature_dim
        self.half_size = half_size
        self.cube = torch.nn.Parameter(torch.ones((1, feature_dim, res, res, res)))
        self.feature_dim = feature_dim
    
    def forward(self, loc):
        # grid_sample(
        #   triplane (N=1, C=dim, D_in=res, H_in=res, W_in=res), 
        #   loc (N=1, D_out=B, H_out=1, W_out=1, 3)
        # ) 
        # -> (N=1, C=dim, D_out=B, H_out=1, W_out=1)
        norm_loc = loc / self.half_size  # (B, 3)
        norm_loc = norm_loc.reshape(1, len(loc), 1, 1, 3)
        sampled = torch.nn.functional.grid_sample(self.cube, norm_loc, align_corners=True, padding_mode="border", mode="bilinear")

        sampled = sampled.reshape(self.dim, len(loc)).permute(1, 0)

        return sampled

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, feature_dim=24, max_freq=10):
        super(PositionalEmbedding, self).__init__()
        self.n_freq = max(feature_dim // (3 * 2), 1)
        self.max_freq = max_freq
        self.feature_dim = 3 * (2 * self.n_freq)
        
    def forward(self, loc):
        freq_bands = 2.0 ** torch.linspace(0, self.max_freq, steps=self.n_freq)
        freq_loc = loc.unsqueeze(2) * freq_bands
        sin_cos = torch.cat((torch.sin(freq_loc), torch.cos(freq_loc)), dim=1)
        
        latent = sin_cos.view(len(loc), self.feature_dim)

        return latent

# (TODO): artifact issues
# class Triplane(torch.nn.Module):
#     def __init__(self, res=256, feature_dim=24, half_size=2):
#         super(Triplane, self).__init__()
#         self.dim = max(feature_dim // 3, 1)
#         self.half_size = half_size
#         self.triplane = torch.nn.Parameter(torch.ones((3, self.dim, res, res)))
#         self.feature_dim = self.dim * 3
    
#     def forward(self, loc):
#         # grid_sample(
#         #   triplane (N=1, C=dim, H_in=res, W_in=res), 
#         #   loc (N=1, H_out=B, W_out=1, 2)
#         # ) 
#         # -> (N=1, C=dim, H_out=B, W_out=1)
#         norm_loc = loc / self.half_size  # (B, 3)
#         xy = norm_loc[:, :2].reshape(1, len(loc), 1, 2)
#         yz = norm_loc[:, 1:3].reshape(1, len(loc), 1, 2)
#         xz = norm_loc[:, [0, 2]].reshape(1, len(loc), 1, 2)

#         sampled_xy = torch.nn.functional.grid_sample(self.triplane[0:1], xy, align_corners=True, padding_mode="border", mode="bicubic")
#         sampled_yz = torch.nn.functional.grid_sample(self.triplane[1:2], yz, align_corners=True, padding_mode="border", mode="bicubic")
#         sampled_xz = torch.nn.functional.grid_sample(self.triplane[2:3], xz, align_corners=True, padding_mode="border", mode="bicubic")

#         sampled_xy = sampled_xy.reshape(self.dim, len(loc)).permute(1, 0)
#         sampled_yz = sampled_yz.reshape(self.dim, len(loc)).permute(1, 0)
#         sampled_xz = sampled_xz.reshape(self.dim, len(loc)).permute(1, 0)

#         latent_vector = torch.cat([sampled_xy, sampled_yz, sampled_xz], dim=1)

#         return latent_vector