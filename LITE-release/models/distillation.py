"""LITE main models (release version).

Contains only the components needed to reproduce the paper's main model:
  * ShuffleNetMultimodalModel / ResTextTeacher  -- multimodal teacher
    (ShuffleNetV2-x2.0 image branch + text branch + cross-attention),
    consuming pre-extracted token-level text features (B, L, 768).
  * ResTextStudent -- image-only student (ShuffleNetV2-x1.0) with the
    width-matched three-layer MLP text simulator (256 -> 620 -> 620 -> 256,
    ~0.71M params). This is the released main model (mlp_matched).
  * MultiLevelDistillationLoss -- three-way feature alignment loss
    (img / text / fused, weights 0.3 / 0.2 / 0.5) + hard-label + soft-label terms.

Checkpoint compatibility: attribute names and layer indices are identical to the
training-time definitions, so released `best.pth` files load with strict=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0, shufflenet_v2_x2_0


# ==================== 多模态骨干（教师内部结构） ====================

class ShuffleNetMultimodalModel(nn.Module):
    """简化的多模态ShuffleNet模型"""
    def __init__(self, num_classes=8, hidden_dim=256,
                 use_cross_attention=True, model_size='x2_0'):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.model_size = model_size

        # 根据model_size选择ShuffleNet版本
        if model_size == 'x2_0':
            self.img_encoder = shufflenet_v2_x2_0(pretrained=True)
        else:
            self.img_encoder = shufflenet_v2_x1_0(pretrained=True)
        self.img_encoder.fc = nn.Identity()

        # 获取特征维度
        with torch.no_grad():
            mock_input = torch.randn(1, 3, 224, 224)
            img_features = self.img_encoder(mock_input)
            self.feature_dim = img_features.shape[1]

        # 图像特征投影
        self.img_proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 文本处理
        self.text_process = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 文本聚合
        self.txt_aggregate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # 跨模态注意力（可选）
        if use_cross_attention:
            self.cross_att = nn.MultiheadAttention(
                hidden_dim, num_heads=4, batch_first=True
            )
            self.cross_att_norm = nn.LayerNorm(hidden_dim)

        # 融合模块：concat(img_proj, txt_pooled) -> MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, text, images, return_features=False):
        # 图像特征
        img_feat = self.img_encoder(images)
        img_feat = self.img_proj(img_feat)

        # 文本特征：对 (B, L, hidden) 沿 token 维做平均（未使用 attention mask，
        # PAD 位置参与平均——与发布权重的训练实现完全一致，如实保留）
        txt_feat = self.text_process(text)
        txt_pooled = self.txt_aggregate(txt_feat.mean(dim=1))

        # 跨模态注意力
        if self.use_cross_attention:
            img_seq = img_feat.unsqueeze(1)
            cross_feat, _ = self.cross_att(
                query=img_seq, key=txt_feat, value=txt_feat
            )
            img_feat = self.cross_att_norm(img_seq + cross_feat)
            img_feat = img_feat.squeeze(1)

        # 融合
        concat = torch.cat([img_feat, txt_pooled], dim=1)
        fused = self.fusion(concat)

        # 分类
        logits = self.classifier(fused)

        if return_features:
            # 与 logits 同源返回中间特征，避免重复前向与潜在数值不一致
            features = {
                'img_features': img_feat,
                'text_features': txt_pooled,
                'fused_features': fused,
            }
            return logits, features
        return logits


class ResTextTeacher(nn.Module):
    """双模态教师模型 - 完整的ResText"""
    def __init__(self, num_classes=8, hidden_dim=256, model_size='x2_0'):
        super().__init__()
        self.backbone = ShuffleNetMultimodalModel(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            use_cross_attention=True,
            model_size=model_size
        )

    def forward(self, text, images, return_features=False):
        # 直接复用 backbone 的同源特征，避免重复前向
        return self.backbone(text, images, return_features=return_features)


# ==================== 主模型学生（width-matched MLP 模拟器） ====================

class ResTextStudent(nn.Module):
    """单模态学生模型 - 仅使用图像。

    文本模拟器固定为 width-matched 三层 MLP（mlp_matched）：256 -> 620 -> 620 -> 256，
    参数量约 0.71M，与多尺度模拟器（约 0.69M）容量对齐；发布 checkpoints
    均以此结构训练。层索引与训练时定义逐层一致，state_dict 严格兼容。
    """
    def __init__(self, num_classes=8, hidden_dim=256, use_text_simulator=True,
                 simulator_type='mlp_matched'):
        super().__init__()
        if simulator_type != 'mlp_matched':
            raise ValueError(
                "release only contains the main model; simulator_type must be "
                f"'mlp_matched' (got {simulator_type!r})")
        self.use_text_simulator = use_text_simulator
        self.simulator_type = simulator_type

        # 使用更轻量的ShuffleNet作为主干
        self.img_encoder = shufflenet_v2_x1_0(pretrained=True)
        self.img_encoder.fc = nn.Identity()

        # 获取特征维度
        with torch.no_grad():
            mock_input = torch.randn(1, 3, 224, 224)
            img_features = self.img_encoder(mock_input)
            feature_dim = img_features.shape[1]

        # 图像特征投影层
        self.img_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # width-matched 三层 MLP 文本模拟器（Linear+GELU+LN+Dropout ×3）
        w = 620
        self.text_simulator = nn.Sequential(
            nn.Linear(hidden_dim, w),
            nn.GELU(),
            nn.LayerNorm(w),
            nn.Dropout(0.1),
            nn.Linear(w, w),
            nn.GELU(),
            nn.LayerNorm(w),
            nn.Dropout(0.1),
            nn.Linear(w, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # 轻量级融合模块
        self.lightweight_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 温度参数 (用于软标签蒸馏)
        self.temperature = 8.0

    def forward(self, images, return_features=False):
        # 图像特征提取
        img_feat = self.img_encoder(images)
        img_feat = self.img_proj(img_feat)

        # 模拟文本特征
        if self.use_text_simulator:
            simulated_text = self.text_simulator(img_feat)
        else:
            simulated_text = torch.zeros_like(img_feat)

        # 轻量级融合
        concat_feat = torch.cat([img_feat, simulated_text], dim=1)
        fused_feat = self.lightweight_fusion(concat_feat)

        # 分类输出
        logits = self.classifier(fused_feat)

        if return_features:
            feats = {
                'img_features': img_feat,
                'fused_features': fused_feat,
            }
            # 有模拟器才返回 simulated_text，使损失中 text 对齐项生效；
            # 无模拟器时不返回该键 -> 损失自动跳过 text 对齐（避免零向量余弦 nan）
            if self.use_text_simulator:
                feats['simulated_text'] = simulated_text
            return logits, feats

        return logits

    def get_soft_targets(self, logits, temperature=None):
        """获取软目标"""
        if temperature is None:
            temperature = self.temperature
        return F.softmax(logits / temperature, dim=1)


# ==================== 蒸馏损失（三路特征对齐） ====================

class MultiLevelDistillationLoss(nn.Module):
    """多层次蒸馏损失：hard + soft + 三路特征对齐（img/text/fused）。

    total = alpha * CE(logits, y)
          + beta * T^2 * KL(student_T || teacher_T)
          + gamma * sum_k w_k * (MSE_k + (1 - cos_k)) / denominator

    feature_weights 默认 {'img_features': 0.3, 'text_features': 0.2,
    'fused_features': 0.5}；lambda_T 消融（--lambda-text 0）时传
    text_features=0.0 且 feature_denominator=1.0（固定分母，不重新归一化）。
    """
    def __init__(self, alpha=0.7, beta=0.1, gamma=0.05, temperature=8.0,
                 feature_weights=None, feature_denominator=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

        # 特征权重配置（默认 img:0.3 / text:0.2 / fused:0.5）
        if feature_weights is None:
            feature_weights = {
                'img_features': 0.3,      # 图像特征
                'text_features': 0.2,     # 文本特征（学生 simulated_text vs 教师 text_features）
                'fused_features': 0.5     # 融合特征（最重要）
            }
        self.feature_weights = feature_weights
        # 归一化分母：None = 配置权重总和；显式传入则固定（消融时固定 1.0）
        self.feature_denominator = feature_denominator

    def feature_alignment_loss(self, student_feat, teacher_feat):
        """特征对齐损失 - L2距离 + 余弦相似度"""
        # L2距离
        l2_loss = self.mse_loss(student_feat, teacher_feat.detach())

        # 余弦相似度损失
        student_flat = student_feat.flatten(1)
        teacher_flat = teacher_feat.detach().flatten(1)

        cosine_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1)
        cosine_loss = (1 - cosine_sim.mean())

        return l2_loss + cosine_loss

    def forward(self, student_logits, teacher_logits, labels,
                student_features=None, teacher_features=None):

        # 1. 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)

        # 2. 软标签蒸馏损失
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # 3. 多层次特征蒸馏损失
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            feature_loss = 0.0
            has_feat = False

            # 对齐图像特征
            if 'img_features' in student_features and 'img_features' in teacher_features:
                loss = self.feature_alignment_loss(
                    student_features['img_features'],
                    teacher_features['img_features']
                )
                feature_loss += self.feature_weights['img_features'] * loss
                has_feat = True

            # 对齐文本特征（学生模型的simulated_text vs 教师模型的text_features）
            if 'simulated_text' in student_features and 'text_features' in teacher_features:
                loss = self.feature_alignment_loss(
                    student_features['simulated_text'],
                    teacher_features['text_features']
                )
                feature_loss += self.feature_weights['text_features'] * loss
                has_feat = True

            # 对齐融合特征
            if 'fused_features' in student_features and 'fused_features' in teacher_features:
                loss = self.feature_alignment_loss(
                    student_features['fused_features'],
                    teacher_features['fused_features']
                )
                feature_loss += self.feature_weights['fused_features'] * loss
                has_feat = True

            # 归一化分母与"实际存在的路"解耦：
            #   None -> 配置权重总和（默认 0.3+0.2+0.5=1.0）
            #   显式传入（lambda_T=0 消融传 1.0）-> 固定基准，恰好多掉该路贡献
            denom = (self.feature_denominator if self.feature_denominator is not None
                     else sum(self.feature_weights.values()))
            if has_feat and denom > 0:
                feature_loss = feature_loss / denom

        # 总损失
        total_loss = (self.alpha * hard_loss +
                      self.beta * soft_loss +
                      self.gamma * feature_loss)

        return total_loss, {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'feature_loss': feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss,
            'total_loss': total_loss.item()
        }


def unit_test():
    """冒烟测试：前向形状、特征契约、损失可反传。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    text = torch.randn(batch_size, 128, 768).to(device)
    labels = torch.randint(0, 8, (batch_size,)).to(device)

    print('1. teacher forward...')
    teacher = ResTextTeacher(num_classes=8).to(device)
    t_logits, t_feat = teacher(text, images, return_features=True)
    print(f'   logits {tuple(t_logits.shape)}  features {list(t_feat.keys())}')

    print('2. student forward (mlp_matched)...')
    student = ResTextStudent(num_classes=8).to(device)
    s_logits, s_feat = student(images, return_features=True)
    params = sum(p.numel() for p in student.parameters())
    assert s_feat['simulated_text'].shape == s_feat['img_features'].shape
    print(f'   logits {tuple(s_logits.shape)}  params {params:,}')

    print('3. distillation loss...')
    criterion = MultiLevelDistillationLoss(alpha=0.7, beta=0.1, gamma=0.05,
                                           temperature=8.0)
    loss, loss_dict = criterion(s_logits, t_logits, labels, s_feat, t_feat)
    loss.backward()
    print(f'   total {loss.item():.4f}  {loss_dict}')
    print('=== smoke test passed ===')


if __name__ == '__main__':
    unit_test()
