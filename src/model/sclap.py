import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from .component.model_utilities import MLPLayers
from .component.seld import EINV2_HTSAT
from .component.htsat import HTSAT_Swin_Transformer

torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = self.w(x) 
        weights = F.softmax(scores, dim=-2) 
        output = torch.sum(x * weights, dim=-2)
        return output

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class ModalityClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ModalityClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, 1)
        )
        
    def forward(self, x, alpha=1.0):
        x = GradientReversalFunction.apply(x, alpha)
        x = self.classifier(x)
        return x
    


class sCLAP(nn.Module):
    def __init__(self, cfg, joint_embed_dim=512, mlp_act='relu'):
        super(sCLAP, self).__init__()

        self.mel_bins = cfg.data.n_mels
        self.sed_in_ch, self.doa_in_ch = 1, 7

        self.cfg = cfg
        self.n_events = getattr(cfg.model, 'n_events', 1)
        self.enable_fusion = cfg.model.fusion.enable
        self.fusion_type = cfg.model.fusion.type
        self.audio_backbone = cfg.model.audio.backbone
        self.text_backbone = cfg.model.text.backbone
        
        if mlp_act == 'relu':
            self.mlp_act = nn.ReLU
        elif mlp_act == 'gelu':
            self.mlp_act = nn.GELU

        ####################### Audio Branch #######################
        self.audio_scalar = nn.ModuleList(
            [nn.BatchNorm2d(self.mel_bins) for _ in range(self.doa_in_ch)])
        self.audio_branch = None

        self.fc_doa = nn.Sequential(
            nn.Linear(joint_embed_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, 3 * self.n_events),
            nn.Tanh()
        )

        ####################### Text Branch #######################
        if self.text_backbone == 'roberta':
            self.text_branch = RobertaModel.from_pretrained('roberta-base')
        else: raise NotImplementedError

        self.text_projection = nn.Sequential(
            nn.Linear(768, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))

        # ============================================================
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_text(self, text):
        if self.text_backbone == 'roberta':
            text_output = self.text_branch(
                input_ids=text['input_ids'], 
                attention_mask=text['attention_mask']
                )['pooler_output']
            text_output = self.text_projection(text_output)
            # text_output = F.normalize(text_output, dim=-1)
        else: raise NotImplementedError
        return text_output
    
    def get_text_embedding(self, data):
        """Get the text embedding from the model"""
    # 强制要求 chunk captions 必须存在
        if 'text_chunk1' not in data or data['text_chunk1'] is None:
            raise KeyError("Missing required field 'text_chunk1' in text data (spatialized caption for chunk1).")
        if 'text_chunk2' not in data or data['text_chunk2'] is None:
            raise KeyError("Missing required field 'text_chunk2' in text data (spatialized caption for chunk2).")

        text_comb_embeds = self.encode_text(data['text_comb'])
        text_sed_embeds = self.encode_text(data['text'])
        chunk1_emb = self.encode_text(data['text_chunk1'])
        chunk2_emb = self.encode_text(data['text_chunk2'])

        return [text_comb_embeds, text_sed_embeds, chunk1_emb, chunk2_emb]

    def encode_audio(self):
        raise NotImplementedError

    def get_audio_embedding(self):
        """Get the audio embedding from the model

        """
        raise NotImplementedError

    def load_pretrained_weights(self):
        raise NotImplementedError


class sCLAP_Single(sCLAP):
    def __init__(self, cfg, joint_embed_dim=512, mlp_act='relu'):
        super(sCLAP_Single, self).__init__(cfg, joint_embed_dim, mlp_act)

        ####################### Audio Branch #######################
        if self.audio_backbone == 'HTSAT':
            self.audio_branch = HTSAT_Swin_Transformer(cfg, self.doa_in_ch, 
                                                       **cfg.model.audio.kwargs)
        else: raise NotImplementedError

        self.audio_projection = nn.Sequential(
            nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
        # # Audio SED
        # self.audio_sed_projection = nn.Sequential(
        #     nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
        #     self.mlp_act(),
        #     nn.Linear(joint_embed_dim, joint_embed_dim))
        # # Audio DOA
        # self.audio_doa_projection = nn.Sequential(
        #     nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
        #     self.mlp_act(),
        #     nn.Linear(joint_embed_dim, joint_embed_dim))

        if cfg.ckpt_path is None:
            audio_ckpt_path = cfg.model.audio.ckpt_path
            if isinstance(audio_ckpt_path, list):
                audio_ckpt_path = audio_ckpt_path[0]
            text_ckpt_path = cfg.model.text.ckpt_path
            self.load_pretrained_weights(audio_ckpt_path, text_ckpt_path)
    
    def encode_audio(self, audio, longer_list=[]):
        return self.audio_branch(audio, longer_list)
    
    def get_audio_embedding(self, data, longer_list=[]):
        """Get the audio embedding from the model

        """
        audio = data['audio4doa']
        # Compute scalar
        audio = audio.transpose(1, 3)
        for nch in range(audio.shape[-1]):
            audio[..., [nch]] = self.audio_scalar[nch](audio[..., [nch]])
        audio = audio.transpose(1, 3)

        audio = self.encode_audio(audio, longer_list)
        audio_embeds = self.audio_projection(audio['embedding'])
        # audio_embeds = F.normalize(audio_embeds, dim=-1)

        return [audio_embeds, audio_embeds, audio_embeds]

        # audio_sed_embeds = self.audio_sed_projection(audio['embedding'])
        # audio_sed_embeds = F.normalize(audio_sed_embeds, dim=-1)
        # audio_doa_embeds = self.audio_doa_projection(audio['embedding'])
        # audio_doa_embeds = F.normalize(audio_doa_embeds, dim=-1)
        
        # return [audio_embeds, audio_sed_embeds, audio_doa_embeds]
    
    def forward(self, audio, text, longer_list=[]):
        """Forward audio and text into the sCLAP"""

        audio_embedding = self.get_audio_embedding(audio, longer_list)
        text_embedding = self.get_text_embedding(text)
        doa = self.fc_doa(audio_embedding[-1])
        if doa.dim() == 2:
            b = doa.shape[0]
            doa = doa.view(b, self.n_events, 3)

        return [audio_embedding, text_embedding, doa]
    
    def load_pretrained_weights(self, audio_path, text_path=None):
        """Load the pretrained weights for the audio and text encoder"""

        if audio_path is None:
            return
        all_keys = list(self.state_dict().keys())
        ckpt = torch.load(audio_path, map_location='cpu')['state_dict']
        if 'ACCDOA' in audio_path or 'SEDDOA' in audio_path: # Single Branch Model {ACCDOA, mACCDOA, SEDDOA}
            print('Loading audio encoder from {}'.format(audio_path))
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            for k, v in self.audio_branch.state_dict().items():
                if any([x in k for x in ['mel_conv2d', 'fusion_model']]): 
                    continue
                if 'audio_branch.' + k in all_keys: 
                    all_keys.remove('audio_branch.' + k) 
                v.data.copy_(ckpt['encoder.' + k])
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['scalar.' + k])
        elif 'HTSAT-fullset' in audio_path: # HTSAT Model
            print('Loading HTSAT model from {}'.format(audio_path))
            ckpt = {k.replace('sed_model.', ''): v for k, v in ckpt.items()}
            for k, v in self.audio_branch.state_dict().items():
                if any([x in k for x in ['mel_conv2d', 'fusion_model']]): 
                    continue
                elif k == 'patch_embed.proj.weight':
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k) 
                    paras = ckpt[k].repeat(1, self.doa_in_ch, 1, 1) / self.doa_in_ch
                    v.data.copy_(paras)
                else:
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k) 
                    v.data.copy_(ckpt[k])
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['bn0.' + k[2:]])
        elif '630k-' in audio_path:
            print('Loading LAION-CLAP audio encoder from {}'.format(audio_path))
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            for key, value in self.state_dict().items():
                if key == 'logit_scale': 
                    value.data.copy_(ckpt['logit_scale_a'])
                elif key == 'audio_branch.patch_embed.proj.weight':
                    paras = ckpt[key].repeat(1, self.doa_in_ch, 1, 1) / self.doa_in_ch
                    value.data.copy_(paras)
                elif 'audio_scalar' in key: continue
                elif 'doa' in key: continue
                else: value.data.copy_(ckpt[key])
                all_keys.remove(key)
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['audio_branch.bn0.' + k[2:]])
        else: ValueError('Unknown audio encoder checkpoint: {}'.format(audio_path))     
                
        for key in all_keys:
            # if 'text_branch' in key: continue
            print(f'{key} not loaded.')

        if text_path is None: return
        if '630k-' in text_path:
            print('Loading LAION-CLAP text encoder from {}'.format(text_path))
        else: ValueError('Unknown text encoder checkpoint: {}'.format(text_path))



#Temporal embedding encoder
class TemporalAudioEncoder(nn.Module):

    def __init__(self, embedding_dim=512, num_temporal_steps=2, num_heads=8, dim_feedforward=2048, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Feed-forward and norm layers per transformer layer
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'q_proj': nn.Linear(embedding_dim, embedding_dim),
                'k_proj': nn.Linear(embedding_dim, embedding_dim),
                'v_proj': nn.Linear(embedding_dim, embedding_dim),
                'out_proj': nn.Linear(embedding_dim, embedding_dim),
                'ln1': nn.LayerNorm(embedding_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embedding_dim, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, embedding_dim),
                    nn.Dropout(dropout),
                ),
                'ln2': nn.LayerNorm(embedding_dim),
                'dropout': nn.Dropout(dropout)
            })
            self.layers.append(layer)

        self.final_ln = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def _get_sin_cos(self, seq_len, device, dtype):
        dim = self.head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        positions = torch.arange(seq_len, device=device).type_as(inv_freq).unsqueeze(1)
        freqs = positions * inv_freq.unsqueeze(0)  # (seq_len, dim/2)
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        # repeat each value twice to match head_dim
        sin = sin.repeat_interleave(2, dim=-1).type(dtype)
        cos = cos.repeat_interleave(2, dim=-1).type(dtype)
        return sin, cos

    def _rotate_every_two(self, x):
        x_shape = x.shape
        x = x.view(*x_shape[:-1], -1, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x_rot = torch.stack((-x2, x1), dim=-1)
        return x_rot.reshape(x_shape)

    def _apply_rope(self, x, sin, cos):
        # x: (B, H, T, head_dim), sin/cos: (T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_every_two(x) * sin

    def _multi_head_self_attention(self, x, layer):
        # x: (B, T, D)
        B, T, D = x.size()
        
        # Project and reshape to (B, H, T, head_dim)
        q = layer['q_proj'](x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = layer['k_proj'](x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = layer['v_proj'](x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        device = x.device
        dtype = x.dtype
        sin, cos = self._get_sin_cos(T, device, dtype)

        q = self._apply_rope(q, sin, cos)
        k = self._apply_rope(k, sin, cos)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = layer['dropout'](attn)
        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = layer['out_proj'](out)
        return out

    def forward(self, audio_embeddings):
        # audio_embeddings: (B, T, D)
        x = audio_embeddings
        for layer in self.layers:
            residual = x
            attn_out = self._multi_head_self_attention(x, layer)
            x = residual + layer['dropout'](attn_out)
            x = layer['ln1'](x)

            residual = x
            ffn_out = layer['ffn'](x)
            x = residual + ffn_out
            x = layer['ln2'](x)

        x = self.final_ln(x)
        x = x.mean(dim=1)
        temporal_embeds = self.mlp(x)
        return temporal_embeds

class sCLAP_Dual(sCLAP):
    def __init__(self, cfg, joint_embed_dim=512, mlp_act='relu'):
        super(sCLAP_Dual, self).__init__(cfg, joint_embed_dim, mlp_act)

        ####################### Audio Branch #######################
        if self.audio_backbone == 'HTSAT':
            self.audio_branch = EINV2_HTSAT(cfg, self.sed_in_ch, self.doa_in_ch)
        else: raise NotImplementedError
        # Audio SED
        self.audio_sed_projection = nn.Sequential(
            nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
        # Audio DOA
        self.audio_doa_projection = nn.Sequential(
            nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
       
        #Audio Temporal
        self.audio_temporal_encoder = TemporalAudioEncoder(
            embedding_dim=joint_embed_dim,
            num_temporal_steps=2,
            num_heads=8,
            dim_feedforward=2048,
            num_layers=2
        )

         #Audio Temporal Projection 
        self.audio_temporal_projection = nn.Sequential(
            nn.Linear(cfg.model.audio.output_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
        
        #Final Audio Projection
        self.final_audio_projection= nn.Sequential(
            nn.Linear(joint_embed_dim, joint_embed_dim),
            self.mlp_act(),
            nn.Linear(joint_embed_dim, joint_embed_dim))
        
        #Modality Classifier
        self.modality_classifier = ModalityClassifier(joint_embed_dim)

        # self.audio_projection = nn.Sequential(
        #     # nn.Linear(cfg.model.audio.output_dim*2, joint_embed_dim),
        #     nn.Linear(joint_embed_dim * 2, joint_embed_dim),
        #     self.mlp_act(),
        #     nn.Linear(joint_embed_dim, joint_embed_dim))

        # ============================================================
        self.weights = nn.Parameter(torch.ones([joint_embed_dim]))

        self.temporal_alpha = nn.Parameter(torch.tensor(1e-4))
        
        self.att_pool_sed = AttentionPooling(cfg.model.audio.output_dim)
        self.att_pool_doa = AttentionPooling(cfg.model.audio.output_dim)

        if cfg.ckpt_path is None: 
            self.load_pretrained_weights(cfg.model.audio.ckpt_path[0], 
                                         cfg.model.audio.ckpt_path[1], 
                                         cfg.model.text.ckpt_path)
        for stitch in self.audio_branch.stitch1:
            stitch.weight.data[:, 0, 0].fill_(1)
            stitch.weight.data[:, 0, 1].zero_()
            stitch.weight.data[:, 1, 0].zero_()
            stitch.weight.data[:, 1, 1].fill_(1)
        self.audio_branch.stitch1.requires_grad_(True)



    def encode_audio(self, audio1, audio2, longer_list=[]):
        return self.audio_branch(audio1, audio2, longer_list)
    
    def _to_duration_pair(self, item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().tolist()
        
        if isinstance(item, (float, int)):
            return float(item), 1.0

        if isinstance(item, (list, tuple)):
            d1 = float(item[0]) if len(item) > 0 else 1.0
            d2 = float(item[1]) if len(item) > 1 else 1.0
            return d1, d2
            
        return 1.0, 1.0

    def _normalize_ori_audio_duration(self, ori_durations, batch_size: int):
        # Case: Single Tensor (B, 2)
        if isinstance(ori_durations, torch.Tensor):
            t = ori_durations.detach().cpu()
            if t.numel() == batch_size * 2:
                t = t.reshape(batch_size, 2)
                return [(float(t[i, 0]), float(t[i, 1])) for i in range(batch_size)]
        
        # Case: List of 3 Tensors (Triplet default collate) -> Interleave
        if isinstance(ori_durations, (list, tuple)) and len(ori_durations) == 3 and batch_size % 3 == 0:
            B0 = batch_size // 3
            c0, c1, c2 = ori_durations
            
            def get_pair(c, idx):
                if isinstance(c, torch.Tensor):
                    return float(c[idx, 0]), float(c[idx, 1])
                elif isinstance(c, (list, tuple)):
                    return self._to_duration_pair(c[idx])
                return 1.0, 1.0

            len0 = len(c0) if hasattr(c0, '__len__') else 0
            
            if len0 == B0:
                out = []
                for i in range(B0):
                    out.append(get_pair(c0, i))
                    out.append(get_pair(c1, i))
                    out.append(get_pair(c2, i))
                return out

        # Case: Already flattened list of correct length
        if isinstance(ori_durations, (list, tuple)) and len(ori_durations) == batch_size:
            return [self._to_duration_pair(x) for x in ori_durations]
            
        return [(1.0, 1.0)] * batch_size

    def get_audio_embedding(self, data, longer_list=[]):
        """Get the audio embedding from the model"""

        audio1, audio2 = data['audio4sed'], data['audio4doa']
        # Compute audio4sed scalar
        audio1 = self.audio_scalar[0](audio1.transpose(1, 3)).transpose(1, 3)
        # Compute audio4doa scalar
        audio2 = audio2.transpose(1, 3)
        for nch in range(audio2.shape[-1]):
            audio2[..., [nch]] = self.audio_scalar[nch](audio2[..., [nch]])
        audio2 = audio2.transpose(1, 3)

        B, C, F, T = audio1.shape
        ori_durations_raw = data.get('ori_audio_duration', None)
        assert ori_durations_raw is not None, "audio dict missing key 'ori_audio_duration'"
        ori_durations = self._normalize_ori_audio_duration(ori_durations_raw, B)
        
        audio1_list = []
        audio2_list = []
        split_indices = []
        
        # 1. Prepare 3xB batch data (Full, Chunk1_Hard, Chunk2_Hard)
        for i in range(B):
            d1, d2 = ori_durations[i]
            total_duration = d1 + d2
            ratio = d1 / total_duration if total_duration > 0 else 0.5
            split_idx = int(T * ratio)
            split_idx = max(1, min(split_idx, T - 1))
            split_indices.append(split_idx)
            
            full_a1, full_a2 = audio1[i], audio2[i]
            
            # Hard Split: Slice and Pad to T
            c1_a1 = F.pad(full_a1[..., :split_idx], (0, T - split_idx))
            c1_a2 = F.pad(full_a2[..., :split_idx], (0, T - split_idx))
            
            # Chunk 2 moved to start for position encoding consistency
            c2_a1 = F.pad(full_a1[..., split_idx:], (0, split_idx))
            c2_a2 = F.pad(full_a2[..., split_idx:], (0, split_idx))
            
            audio1_list.extend([full_a1, c1_a1, c2_a1])
            audio2_list.extend([full_a2, c1_a2, c2_a2])

        # Batch Encoding (3B samples)
        audio1_batch = torch.stack(audio1_list, dim=0)
        audio2_batch = torch.stack(audio2_list, dim=0)
        
        if isinstance(longer_list, torch.Tensor) and longer_list.numel() > 0:
             longer_ids = longer_list.detach().cpu().tolist()
             expanded_longer = torch.tensor([3 * idx for idx in longer_ids], device=longer_list.device)
        else:
            expanded_longer = []

        audio_output = self.encode_audio(audio1_batch, audio2_batch, expanded_longer)

        # 2. Extract All Global Embeddings
        all_sed_embs = self.audio_sed_projection(audio_output['sed_embedding'])
        all_doa_embs = self.audio_doa_projection(audio_output['doa_embedding'])
        
        def get_joint_emb(sed, doa):
            return sed + self.weights * doa

        all_audio_embs = get_joint_emb(all_sed_embs, all_doa_embs)
        
        full_idx = torch.arange(0, 3*B, 3, device=all_audio_embs.device)
        hard1_idx = torch.arange(1, 3*B, 3, device=all_audio_embs.device)
        hard2_idx = torch.arange(2, 3*B, 3, device=all_audio_embs.device)
        
        # Main Outputs
        audio_embeds = all_audio_embs[full_idx]
        audio_sed_embeds = all_sed_embs[full_idx]
        audio_doa_embeds = all_doa_embs[full_idx]
        
        # Hard Chunk Embeds
        chunck1_embeds_hard = all_audio_embs[hard1_idx].detach()
        chunck2_embeds_hard = all_audio_embs[hard2_idx].detach()
    
        # 3. Soft Slicing (From FULL branch feature maps)
        sed_feature_maps = audio_output['sed_feature_maps'][full_idx]
        doa_feature_maps = audio_output['doa_feature_maps'][full_idx]

        chunck1_embeds_list = []
        chunck2_embeds_list = []
        
        for i in range(B):
            split_idx = split_indices[i]
            
            # Soft Chunk 1
            x_s1 = sed_feature_maps[i, ..., :split_idx].mean(dim=1).transpose(0, 1) # (T1, C)
            x_d1 = doa_feature_maps[i, ..., :split_idx].mean(dim=1).transpose(0, 1) # (T1, C)
            emb1 = get_joint_emb(self.audio_sed_projection(self.att_pool_sed(x_s1)), 
                                 self.audio_doa_projection(self.att_pool_doa(x_d1)))

            # Soft Chunk 2
            x_s2 = sed_feature_maps[i, ..., split_idx:].mean(dim=1).transpose(0, 1) # (T2, C)
            x_d2 = doa_feature_maps[i, ..., split_idx:].mean(dim=1).transpose(0, 1) # (T2, C)
            emb2 = get_joint_emb(self.audio_sed_projection(self.att_pool_sed(x_s2)), 
                                 self.audio_doa_projection(self.att_pool_doa(x_d2)))
                
            chunck1_embeds_list.append(emb1)
            chunck2_embeds_list.append(emb2)
        
        chunck1_embeds = torch.stack(chunck1_embeds_list, dim=0)
        chunck2_embeds = torch.stack(chunck2_embeds_list, dim=0)
        
        temporal_seq = torch.stack([chunck1_embeds, chunck2_embeds], dim=1)
        audio_temporal_embeds = self.audio_temporal_encoder(temporal_seq) 

        audio_triplet_embeds = audio_embeds + self.temporal_alpha * self.final_audio_projection(audio_temporal_embeds)
        
        return [audio_embeds, audio_sed_embeds, audio_doa_embeds, 
                audio_temporal_embeds, audio_triplet_embeds, 
                chunck1_embeds, chunck2_embeds, 
                chunck1_embeds_hard, chunck2_embeds_hard]

    def forward(self, audio, text, longer_list=[]):
        """Forward audio and text into the sCLAP"""

        audio_embedding = self.get_audio_embedding(audio, longer_list)
        text_embedding = self.get_text_embedding(text)

        audio_embedding = [F.normalize(x, dim=-1) for x in audio_embedding]
        text_embedding = [F.normalize(x, dim=-1) for x in text_embedding]
        
        # Use audio_triplet_embeds (index 4) for DOA prediction
        doa = self.fc_doa(audio_embedding[4]) 
        if doa.dim() == 2:
            b = doa.shape[0]
            doa = doa.view(b, self.n_events, 3)

        return [audio_embedding, text_embedding, doa]
    
    def load_pretrained_weights(self, audio_path1, audio_path2, text_path):
        """Load the pretrained weights for the audio and text encoder

        Parameters
        ----------
        audio_path: str
            the path to the audio encoder pretrained weights
        text_path: str
            the path to the text encoder pretrained weights
        seld_path: str
            the path to the PSELDNets pretrained weights
        """
        if audio_path1 is None and audio_path2 is None:
            return

        all_keys = list(self.state_dict().keys())
        # Load pseldnets-EINV2 first 
        if audio_path1 and 'EINV2' in audio_path1:
            ckpt = torch.load(audio_path1, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            print('Loading PSELDNets pretrained weights from ', audio_path1)
            for k, v in self.audio_branch.state_dict().items():
                if k == 'sed_encoder.patch_embed.proj.weight':
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k)
                    paras = ckpt[k][:, :v.shape[1]] * 4
                    v.data.copy_(paras)
                elif any([x in k for x in ['mel_conv2d', 'fusion_model']]): 
                    continue
                else:
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k) 
                    v.data.copy_(ckpt[k])
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['scalar.' + k])
        elif audio_path2 and 'ACCDOA' in audio_path2:
            ckpt = torch.load(audio_path2, map_location='cpu')['state_dict']
            ckpt = {k.replace('net.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()} # if compiling the model
            print('Loading PSELDNets pretrained weights from ', audio_path2)
            for k, v in self.audio_branch.state_dict().items():
                if 'sed_encoder' in k: continue
                elif any([x in k for x in ['mel_conv2d', 'fusion_model']]): 
                    continue
                elif 'doa_encoder' in k:
                    if 'audio_branch.' + k in all_keys: 
                        all_keys.remove('audio_branch.' + k)
                    v.data.copy_(ckpt[k[4:]])
            for k, v in self.audio_scalar.state_dict().items():
                if 'audio_scalar.' + k in all_keys: 
                    all_keys.remove('audio_scalar.' + k)
                v.data.copy_(ckpt['scalar.' + k])

        # Load the audio encoder from clap
        if audio_path1 and '630k-' in audio_path1:
            print('Loading LAION-CLAP audio encoder from {}'.format(audio_path1))
            ckpt = torch.load(audio_path1, map_location='cpu')['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            self.logit_scale.data.copy_(ckpt['logit_scale_a'])
            all_keys.remove('logit_scale')
            # Reload the audio encoder from LAION-CLAP
            for k, v in self.audio_branch.sed_encoder.state_dict().items():
                if 'audio_branch.sed_encoder.' + k in all_keys: 
                    all_keys.remove('audio_branch.sed_encoder.' + k)
                v.data.copy_(ckpt['audio_branch.' + k])
            # Load 'audio_projection' from LAION-CLAP into 'audio_sed_projection', 'audio_doa_projection'
            for k, v in self.audio_sed_projection.state_dict().items():
                if 'audio_sed_projection.' + k in all_keys: 
                    all_keys.remove('audio_sed_projection.' + k)
                v.data.copy_(ckpt['audio_projection.' + k])
            for k, v in self.audio_doa_projection.state_dict().items():
                if 'audio_doa_projection.' + k in all_keys: 
                    all_keys.remove('audio_doa_projection.' + k)
                v.data.copy_(ckpt['audio_projection.' + k])
            # Load the text encoder from LAION-CLAP
            for k, v in self.state_dict().items():
                if k in ckpt and 'audio_projection' not in k:
                    if k in all_keys: all_keys.remove(k)
                    v.data.copy_(ckpt[k])
        else: ValueError('Unknown audio encoder checkpoint: {}'.format(audio_path1))

        for key in all_keys:
            # if 'text_branch' in key: continue
            print(f'{key} not loaded.')

        if text_path is None: return
        if audio_path1 and '630k-' in text_path:
            print('Loading LAION-CLAP text encoder from {}'.format(text_path))
        else: ValueError('Unknown text encoder checkpoint: {}'.format(text_path))
