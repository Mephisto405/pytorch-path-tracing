import torch

# Learned embeddings
class Cube(torch.nn.Module):
    def __init__(self, res=30, feature_dim=10, half_size=2):
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
        sampled = torch.nn.functional.grid_sample(
            self.cube, 
            norm_loc, 
            padding_mode="border", 
            mode="bilinear"
        )

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
        sin_cos = torch.concat((torch.sin(freq_loc), torch.cos(freq_loc)), dim=1)
        
        latent = sin_cos.view(len(loc), self.feature_dim)

        return latent

class Triplane(torch.nn.Module):
    """
    show_image(scene["caches"].embed.triplane[0,:3].permute(1, 2, 0), 64, 1, "XY")
    """

    def __init__(self, res=64, feature_dim=64, half_size=2):
        super(Triplane, self).__init__()
        self.dim = max(feature_dim // 3, 1)
        self.half_size = half_size
        self.triplane = torch.nn.Parameter(torch.ones((3, self.dim, res, res)))
        self.feature_dim = self.dim * 3
    
    def forward(self, loc):
        # grid_sample(
        #   triplane (N=3, C=dim, H_in=res, W_in=res), 
        #   loc (N=3, H_out=B, W_out=1, 2)
        # ) 
        # -> (N=3, C=dim, H_out=B, W_out=1)
        norm_loc = loc / self.half_size  # (B, 3)
        xy = norm_loc[:, :2].reshape(1, len(loc), 1, 2)
        yz = norm_loc[:, 1:3].reshape(1, len(loc), 1, 2)
        xz = norm_loc[:, [0, 2]].reshape(1, len(loc), 1, 2)

        indices = torch.concat((xy, yz, xz), dim=0)
        latent_vector = torch.nn.functional.grid_sample(
            self.triplane, 
            indices, 
            padding_mode="border", 
            mode="bilinear"
        )
        latent_vector = latent_vector.permute(2, 0, 1, 3).reshape(len(loc), 3 * self.dim)

        return latent_vector