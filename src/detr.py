
import math
import torch
import torch.nn as nn


class PositionEmbeddingSine2D(nn.Module):
    """
    2D sine positional encoding for DETR encoder input.
    Transformers used for detection need 2D positional encodings added to the
    encoder input (and query positional embeddings added to decoder input).
    Without them the model cannot learn spatial relationships.

    Note: This implementation differs from the original DETR, which is a mask-aware
    coordinate generation that handles padded inputs. This version is good for
    fixed-size inputs without padding.
    """

    def __init__(self, num_pos_feats=128, temperature=10000):
        """Initialize the 2D sine positional encoding module."""
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, features):
        """
        Forward.

        Parameters
        ----------
        features : torch.Tensor
            Input feature map of shape [B, C, H, W].
        Returns
        -------
        pos : torch.Tensor
            Positional embedding of shape [B, 2*num_pos_feats, H, W].
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

    def __init__(
            self,
            backbone: torch.nn.Module,
            backbone_channels: int,
            num_classes: int,
            hidden_dim: int = 256,
            num_queries: int = 100
        ):
        """Initialize the DETR (DEtection TRansformer) model.

        Parameters
        ----------
        backbone : nn.Module
            Feature extractor (for example DINOv2 or ResNet). The forward pass is
            expected to return a list/sequence of feature maps, and the last map is
            used as encoder input.
        backbone_channels : int
            Number of channels in the selected backbone feature map.
        num_classes : int
            Number of foreground classes. The classifier head internally predicts
            ``num_classes + 1`` logits to include the no-object class.
        hidden_dim : int, optional
            Transformer embedding dimension (``d_model``), by default 256.
        num_queries : int, optional
            Number of learned object queries (decoder slots), by default 100.
        """

        super().__init__()
        
        # Use frozen batchnorm for ResNet50?
        # DINOv2 uses LayerNorm, so no need to freeze batchnorm layers.
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        
        # Project backbone output channels to DETR hidden dimension (reduce the size of
        # the channels).
        self.input_proj = nn.Conv2d(backbone_channels, hidden_dim, kernel_size=1)
        
        # Positional embeddings
        # Since Transformers do not inherently possess spatial understanding, DETR adds
        # positional encodings to the output of the backbone. These positional encodings
        # inform the model about the spatial relationships between different parts of the
        # image. The encodings are crucial for the Transformer to understand the absolute
        # and relative positions of objects.
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2)
        
        # DETR transformer encoder/decoder.
        # Using explicit modules lets us expose intermediate decoder states for
        # auxiliary decoder supervision during training.
        self.num_layers = 6

        # The Encoder takes a spatial feature map from the backbone and transforms it into
        # a sequence of context-aware visual tokens using multi-head self-attention. Its
        # role is to make every spatial location aware of every other location in the image.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,  # 1024 is lightweight, 2048 is the default
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        # The Decoder takes a set of learned object queries and the encoder’s output, and
        # transforms each query into a prediction of one object (bounding box + class).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,  # 1024 is lightweight, 2048 is the default
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers,
        )
        
        # Learned object queries (content) and positional embeddings (for decoder).
        self.query_content = nn.Parameter(torch.randn(num_queries, hidden_dim)) # requires_grad=True
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        nn.init.xavier_uniform_(self.query_content)
        
        # Prediction heads (Feed Forward Networks).
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object" class
        self.bbox_embed = nn.Linear(hidden_dim, 4)
        self._logged_shapes = False


    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input image tensor of shape [B, 3, H, W].

        Returns:
        -------
        dict
            A dictionary containing:
            - 'pred_logits': Tensor of shape [B, num_queries, num_classes + 1] with
            class logits for each query.
            - 'pred_boxes': Tensor of shape [B, num_queries, 4] with predicted
            bounding boxes (normalized [0, 1]).
            - 'aux_outputs': List of dictionaries containing intermediate decoder
            outputs for auxiliary loss.   
        """

        # 1. Backbone
        # x: [B, 3, 518, 518] input image tensor
        features = self.backbone(x)[-1]
        h = self.input_proj(features)

        if not self._logged_shapes:
            print(f"Backbone feature shape: {features.shape}")
            print(f"Projected feature shape: {h.shape}")
            self._logged_shapes = True
        
        # Note:
        #  - DINOv2 uses patch size 14, so if the input is 518x518, the feature map
        #    will be 37x37 (518/14 = 37).
        #  - DINOv2 outputs 768-dimensional embeddings per patch by default

        # Compute positional embeddings
        pos = self.pos_embed(h)  # [B, hidden_dim, 37, 37] - same shape as `h`
        
        # Flatten the feature vector for the transformer.
        B, _, _, _ = h.shape                        # [B, hidden_dim, 37, 37]
        h_flat = h.flatten(2).permute(0, 2, 1)      # [B, 37*37, hidden_dim]
        pos_flat = pos.flatten(2).permute(0, 2, 1)  # [B, 37*37, hidden_dim]
        
        # Add positional embeddings to encoder input
        src = h_flat + pos_flat  # [B, 37*37, hidden_dim]
        
        # Prepare decoder input: query content + query positional embeddings.
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]
        query_obj = self.query_content.unsqueeze(0).repeat(B, 1, 1)       # [B, num_queries, hidden_dim]
        
        # 2. Transformer encoder.
        memory = self.transformer_encoder(src)  # [B, 37*37, hidden_dim]

        # 3. Transformer decoder with intermediate outputs for auxiliary loss.
        hidden = query_obj + query_pos
        decoder_states = []
        for layer in self.transformer_decoder.layers:
            hidden = layer(hidden, memory)
            decoder_states.append(hidden)

        if not decoder_states:
            decoder_states.append(hidden)

        if self.transformer_decoder.norm is not None:
            decoder_states[-1] = self.transformer_decoder.norm(decoder_states[-1])

        class_outputs = [self.class_embed(state) for state in decoder_states]
        box_outputs = [self.bbox_embed(state).sigmoid() for state in decoder_states]

        final_class = class_outputs[-1]
        final_boxes = box_outputs[-1]

        aux_outputs = []
        for class_out, box_out in zip(class_outputs[:-1], box_outputs[:-1]):
            aux_outputs.append(
                {
                    "pred_logits": class_out,
                    "pred_boxes": box_out,
                }
            )

        return {
            "pred_logits": final_class,
            "pred_boxes": final_boxes,
            "aux_outputs": aux_outputs,
        }
