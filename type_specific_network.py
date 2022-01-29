import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TypeSpecificNet(nn.Module):
    def __init__(self, args, embeddingnet, n_conditions):
        """ args: Input arguments from the main script
            embeddingnet: The network that projects the inputs into an embedding of embedding_size
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            n_conditions: Integer defining number of different similarity notions
        """
        super().__init__()
        # Boolean indicating whether masks are learned or fixed
        learnedmask = args.learned

        # Boolean indicating whether masks are initialized in equally sized disjoint
        # sections or random otherwise
        prein = args.prein

        # Indicates that there isn't a 1:1 relationship between type specific spaces
        # and pairs of items categories
        if args.rand_typespaces:
            n_conditions = int(np.ceil(n_conditions / float(args.num_rand_embed)))

        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet

        # When true a fully connected layer is learned to transform the general
        # embedding to the type specific embedding
        self.fc_masks = args.use_fc

        # When true we l2 normalize the output type specific embeddings
        self.l2_norm = args.l2_embed

        if self.fc_masks:
            # learn a fully connected layer rather than a mask to project the general embedding
            # into the type specific space
            self.masks = nn.ModuleList(
                [nn.Linear(args.dim_embed, args.dim_embed) for _ in range(n_conditions)]
            )
        else:
            # create the mask
            if learnedmask:
                if prein:
                    # define masks
                    mask_array = torch.full((n_conditions, args.dim_embed), 0.1)
                    mask_len = int(args.dim_embed / n_conditions)
                    for i in range(n_conditions):
                        mask_array[i, i * mask_len : (i + 1) * mask_len] = 1
                    self.masks = torch.nn.parameter.Parameter(mask_array)
                else:
                    # define masks with gradients
                    self.masks = torch.nn.parameter.Parameter(
                        torch.normal(0.9, 0.7, size=(n_conditions, args.dim_embed))
                    )
            else:
                # define masks
                # initialize masks
                mask_array = torch.zeros(
                    (n_conditions, args.dim_embed), requires_grad=False
                )
                mask_len = int(args.dim_embed / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i * mask_len : (i + 1) * mask_len] = 1
                # no gradients for the masks
                self.masks = torch.nn.parameter.Parameter(
                    mask_array, requires_grad=False
                )

    def forward(self, x, c=None):
        """ x: input image data
            c: type specific embedding to compute for the images, returns all embeddings
               when None including the general embedding concatenated onto the end
        """
        embedded_x = self.embeddingnet(x)
        if c is None:
            # used during testing, wants all type specific embeddings returned for an image
            if self.fc_masks:
                masked_embedding = []
                for mask in self.masks:
                    masked_embedding.append(mask(embedded_x).unsqueeze(1))

                masked_embedding = torch.cat(masked_embedding, 1)
                embedded_x = embedded_x.unsqueeze(1)
            else:
                masks = self.masks
                masks = masks.unsqueeze(0).repeat(embedded_x.size(0), 1, 1)
                embedded_x = embedded_x.unsqueeze(1)
                masked_embedding = embedded_x.expand_as(masks) * masks

            if self.l2_norm:
                masked_embedding = F.normalize(
                    masked_embedding, p=2.0, dim=1, eps=1e-10
                )

            return torch.cat((masked_embedding, embedded_x), 1)

        if self.fc_masks:
            mask_norm = 0.0
            masked_embedding = []
            for embed, condition in zip(embedded_x, c):
                mask = self.masks[condition]
                masked_embedding.append(mask(embed.unsqueeze(0)))
                mask_norm += mask.weight.norm(1)

            masked_embedding = torch.cat(masked_embedding)
        else:
            mask = self.masks[c]
            if self.learnedmask:
                mask = F.relu(mask)

            masked_embedding = embedded_x * mask
            mask_norm = mask.norm(1)

        embed_norm = embedded_x.norm(2)
        if self.l2_norm:
            masked_embedding = F.normalize(masked_embedding, p=2.0, dim=1, eps=1e-10)

        return masked_embedding, mask_norm, embed_norm, embedded_x
