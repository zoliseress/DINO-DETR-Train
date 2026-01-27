
import math
import torch
import torch.nn as nn


class PositionEmbeddingSine2D(nn.Module):
    """2D sine positional encoding for DETR encoder input.
       Transformers used for detection need 2D positional encodings added to the
       encoder input (and query positional embeddings added to decoder input).
       Without them the model cannot learn spatial relationships.
    """

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] tensor
        Returns:
            pos: [B, 2*num_pos_feats, H, W] positional embedding
        """
        B, _, H, W = features.shape
        device = features.device

        # Create 2D coordinate grid (y: [H, W], x: [H, W]).
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        # Normalize coordinates to [0, 2π].
        # Using (H-1) and (W-1) matches original DETR scaling.
        y = y / max(H - 1, 1) * 2 * math.pi
        x = x / max(W - 1, 1) * 2 * math.pi

        # Create frequency bands for sine/cosine pairs.
        # dim_t[k] = temperature^(2 * floor(k/2) / num_pos_feats)
        dim_t = torch.arange(self.num_pos_feats, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Divide coordinates by frequency bands → [H, W, num_pos_feats].
        pos_x = x[..., None] / dim_t
        pos_y = y[..., None] / dim_t

        # Apply sine to even indices, cosine to odd indices,
        # then flatten last two dims into a single feature axis.
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=-1
        ).flatten(-2)

        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
            dim=-1
        ).flatten(-2)

        # Concatenate Y and X encodings along feature dim.
        # → shape [H, W, 2*num_pos_feats]
        pos = torch.cat((pos_y, pos_x), dim=-1)

        # Rearrange to [1, 2*num_pos_feats, H, W].
        pos = pos.permute(2, 0, 1).unsqueeze(0)

        # Expand across batch without allocating extra memory.
        return pos.expand(B, -1, -1, -1)


class DETR(nn.Module):

    def __init__(self, backbone, backbone_channels, num_classes, hidden_dim=256, num_queries=100):
        """
        Initialize the DETR (DEtection TRansformer) model.

        Args:
            backbone (nn.Module): Feature-extractor (e.g. DINOv2 backbone). Expected to return
                a sequence/list of feature maps; the last feature map is used as encoder input.
            backbone_channels (int): Number of channels in the backbone feature map that
                will be projected to `hidden_dim` by `input_proj`.
            num_classes (int): Number of object classes (does not include the special
                "no object" class). The classification head outputs `num_classes + 1`
                logits to account for the "no object" class.
            hidden_dim (int, optional): Transformer embedding dimensionality (d_model). Default: 256.
            num_queries (int, optional): Number of learned object queries (decoder slots). Default: 100.

        Attributes:
            backbone (nn.Module): backbone module.
            input_proj (nn.Conv2d): 1x1 conv projecting backbone channels -> hidden_dim.
            pos_embed (PositionEmbeddingSine2D): 2D sine positional encoder used for the encoder input.
            transformer (nn.Transformer): encoder/decoder transformer.
            query_embed (nn.Embedding): learned query positional embeddings.
            query_content (nn.Parameter): learned query content vectors used as decoder input.
            class_embed (nn.Linear): classification head (outputs num_classes + 1 logits).
            bbox_embed (nn.Linear): bbox regression head (4 numbers per query, normalized via sigmoid).
        """

        super().__init__()
        
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        
        # Project DINOv2 output channels -> DETR hidden dimension
        self.input_proj = nn.Conv2d(backbone_channels, hidden_dim, kernel_size=1)
        
        # Positional embeddings
        # Since Transformers do not inherently possess spatial understanding, DETR adds
        # positional encodings to the output of the backbone. These positional encodings
        # inform the model about the spatial relationships between different parts of the
        # image. The encodings are crucial for the Transformer to understand the absolute
        # and relative positions of objects.
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2)
        
        # DETR transformer encoder/decoder
        # Encoder:
        # It takes a spatial feature map from the backbone and transforms it into a sequence
        # of context-aware visual tokens using multi-head self-attention. Its role is to make
        # every spatial location aware of every other location in the image.
        # Decoder:
        # It takes a set of learned object queries and the encoder’s output, and transforms
        # each query into a prediction of one object (bounding box + class).
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,  # this is the default value
            dropout=0.1,
        )
        
        # Learned object queries (content) and positional embeddings (for decoder).
        self.query_content = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        nn.init.xavier_uniform_(self.query_content)
        
        # Prediction heads (Feed Forward Networks).
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object" class
        self.bbox_embed = nn.Linear(hidden_dim, 4)


    def forward(self, x):

        # 1. Backbone
        # x: [B, 3, 518, 518] input image tensor
        features = self.backbone(x)[-1]         # [B, 768, 37, 37]
        h = self.input_proj(features)           # [B, hidden_dim, 37, 37]
        
        # Note:
        #  - DINOv2 uses patch size 14, so if the input is 518x518, the feature map
        #    will be 37x37 (518/14 = 37).
        #  - DINOv2 outputs 768-dimensional embeddings per patch by default

        # Compute positional embeddings
        pos = self.pos_embed(h)  # [B, hidden_dim, 37, 37]
        
        # Flatten the feature vector for the transformer.
        B, _, _, _ = h.shape                        # [B, hidden_dim, 37, 37]
        h_flat = h.flatten(2).permute(2, 0, 1)      # [37*37, B, hidden_dim]
        pos_flat = pos.flatten(2).permute(2, 0, 1)  # [37*37, B, hidden_dim]
        
        # Add positional embeddings to encoder input
        src = h_flat + pos_flat  # [37*37, B, hidden_dim]
        
        # Prepare decoder input: query content + query positional embeddings
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, hidden_dim]
        query_obj = self.query_content.unsqueeze(1).repeat(1, B, 1)       # [num_queries, B, hidden_dim]
        
        # 2. Transformer (encoder + decoder), where batch_first=False.
        hidden = self.transformer(
            src=src,                   # encoder input with positional encoding
            tgt=query_obj + query_pos  # decoder input: query content + query positional embeddings
        )  # output: [num_queries, B, hidden_dim]
        
        # Feed Forward Networks for predictions.
        outputs_class = self.class_embed(hidden)           # [num_queries, B, num_classes+1]
        outputs_boxes = self.bbox_embed(hidden).sigmoid()  # normalized [0–1] boxes
        
        return {
            "pred_logits": outputs_class.transpose(0, 1),    # [B, num_queries, num_classes+1]
            "pred_boxes": outputs_boxes.transpose(0, 1)      # [B, num_queries, 4]
        }
