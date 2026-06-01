
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


class ConditionalDecoderLayer(nn.Module):
    """One decoder block for a lightweight Conditional DETR variant."""

    def __init__(self, hidden_dim: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.spatial_2d_coords = nn.Linear(hidden_dim, 2)
        self.displacement_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lambda_q = nn.Parameter(torch.ones(hidden_dim))

    def _positional_encoding_from_ref_points(self, ref_points: torch.Tensor) -> torch.Tensor:
        """Convert reference points [B,Q,2] to sine-cosine embeddings [B,Q,D]."""
        ref_points = torch.sigmoid(ref_points)

        half_dim = self.lambda_q.shape[0] // 2
        dim_t = torch.arange(half_dim, dtype=torch.float32, device=ref_points.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / max(half_dim, 1))

        pos_x = ref_points[..., 0, None] / dim_t
        pos_y = ref_points[..., 1, None] / dim_t

        pos_x = torch.cat([torch.sin(pos_x[:, :, 0::2]), torch.cos(pos_x[:, :, 1::2])], dim=-1)
        pos_y = torch.cat([torch.sin(pos_y[:, :, 0::2]), torch.cos(pos_y[:, :, 1::2])], dim=-1)

        return torch.cat([pos_x, pos_y], dim=-1)

    def forward(
        self,
        decoder_embeddings: torch.Tensor,
        object_queries: torch.Tensor,
        encoder_output_feats: torch.Tensor,
    ):
        learnable_displacements = self.displacement_ffn(decoder_embeddings)

        self_attn_key = decoder_embeddings + object_queries
        self_attn_query = decoder_embeddings + object_queries
        self_attn_output = self.self_attn(self_attn_query, self_attn_key, decoder_embeddings)[0]
        decoder_embeddings = self.norm1(decoder_embeddings + self.dropout(self_attn_output))

        ref_coords_2d = self.spatial_2d_coords(object_queries)
        ref_points_embed = self._positional_encoding_from_ref_points(ref_coords_2d)

        spatial_query = learnable_displacements * (self.lambda_q * ref_points_embed)

        cross_query = decoder_embeddings + spatial_query
        cross_attn_output = self.cross_attn(cross_query, encoder_output_feats, encoder_output_feats)[0]
        decoder_embeddings = self.norm2(decoder_embeddings + self.dropout(cross_attn_output))

        ffn_out = self.ffn(decoder_embeddings)
        decoder_embeddings = self.norm3(decoder_embeddings + self.dropout(ffn_out))

        reference_point = torch.cat([ref_coords_2d, torch.zeros_like(ref_coords_2d)], dim=-1)
        return decoder_embeddings, reference_point


class DETR(nn.Module):

    def __init__(
            self,
            backbone: torch.nn.Module,
            backbone_channels: int,
            num_classes: int,
            hidden_dim: int = 256,
            num_queries: int = 100,
            use_conditional_decoder: bool = False,
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
        use_conditional_decoder : bool, optional
            If True, uses a Conditional DETR-style decoder instead of the default
            Transformer decoder stack.
        """

        super().__init__()
        
        # Use frozen batchnorm for ResNet50?
        # DINOv2 uses LayerNorm, so no need to freeze batchnorm layers.
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.use_conditional_decoder = bool(use_conditional_decoder)
        
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
        if self.use_conditional_decoder:
            self.decoder_layers = nn.ModuleList(
                [ConditionalDecoderLayer(hidden_dim=hidden_dim, n_heads=8, dropout=0.1)
                 for _ in range(self.num_layers)]
            )
            self.transformer_decoder = None
        else:
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
            self.decoder_layers = None
        
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

        # 3. Decoder with intermediate outputs for auxiliary loss.
        decoder_states = []
        reference_points = []

        if self.use_conditional_decoder:
            object_queries = query_obj + query_pos
            hidden = object_queries

            for layer in self.decoder_layers:
                hidden, ref_point = layer(hidden, object_queries, memory)
                decoder_states.append(hidden)
                reference_points.append(ref_point)
        else:
            hidden = query_obj + query_pos
            for layer in self.transformer_decoder.layers:
                hidden = layer(hidden, memory)
                decoder_states.append(hidden)

            if self.transformer_decoder.norm is not None:
                decoder_states[-1] = self.transformer_decoder.norm(decoder_states[-1])

        if not decoder_states:
            decoder_states.append(hidden)

        class_outputs = [self.class_embed(state) for state in decoder_states]
        if self.use_conditional_decoder:
            box_outputs = [
                torch.sigmoid(self.bbox_embed(state) + ref_point)
                for state, ref_point in zip(decoder_states, reference_points)
            ]
        else:
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
