from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import CLIPModel


@dataclass
class CLIPContrastiveOutput:
    loss: Optional[Tensor]
    logits_per_image: Optional[Tensor]
    logits_per_text: Optional[Tensor]
    image_embeds: Optional[Tensor]
    text_embeds: Optional[Tensor]
    logit_scale: Tensor


class CLIPContrastiveModel(nn.Module):
    def __init__(
        self,
        clip: CLIPModel,
        normalize: bool = True,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.clip = clip
        self.normalize = normalize
        self.max_logit_scale = max_logit_scale

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        normalize: bool = True,
        max_logit_scale: float = 100.0,
        train_vision: bool = True,
        train_text: bool = True,
        train_projection: bool = True,
        train_logit_scale: bool = True,
        **kwargs,
    ) -> "CLIPContrastiveModel":
        clip = CLIPModel.from_pretrained(model_name_or_path, **kwargs)
        model = cls(
            clip=clip,
            normalize=normalize,
            max_logit_scale=max_logit_scale,
        )
        model.set_trainable(
            train_vision=train_vision,
            train_text=train_text,
            train_projection=train_projection,
            train_logit_scale=train_logit_scale,
        )
        return model

    def set_trainable(
        self,
        train_vision: bool = True,
        train_text: bool = True,
        train_projection: bool = True,
        train_logit_scale: bool = True,
    ) -> None:
        self._set_module_grad(self.clip.vision_model, train_vision)
        self._set_module_grad(self.clip.text_model, train_text)
        self._set_module_grad(self.clip.visual_projection, train_projection)
        self._set_module_grad(self.clip.text_projection, train_projection)
        self.clip.logit_scale.requires_grad = train_logit_scale

    @staticmethod
    def _set_module_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def encode_image(
        self,
        pixel_values: Tensor,
        normalize: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Tensor:
        image_embeds = self.clip.get_image_features(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        return self._maybe_normalize(image_embeds, normalize)

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        normalize: Optional[bool] = None,
    ) -> Tensor:
        text_embeds = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self._maybe_normalize(text_embeds, normalize)

    def compute_similarity(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logit_scale = self.clip.logit_scale.exp().clamp(max=self.max_logit_scale)
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, logit_scale

    def contrastive_loss(
        self,
        logits_per_image: Tensor,
        logits_per_text: Optional[Tensor] = None,
    ) -> Tensor:
        if logits_per_text is None:
            logits_per_text = logits_per_image.t()

        labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        return 0.5 * (image_loss + text_loss)

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        interpolate_pos_encoding: bool = False,
        return_loss: bool = True,
        return_dict: bool = True,
    ) -> CLIPContrastiveOutput | tuple[
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Tensor,
    ]:
        image_embeds = None
        text_embeds = None
        logits_per_image = None
        logits_per_text = None
        loss = None

        if pixel_values is not None:
            image_embeds = self.encode_image(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

        if input_ids is not None:
            text_embeds = self.encode_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        logit_scale = self.clip.logit_scale.exp().clamp(max=self.max_logit_scale)

        if image_embeds is not None and text_embeds is not None:
            logits_per_image, logits_per_text, logit_scale = self.compute_similarity(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
            )
            if return_loss:
                if image_embeds.size(0) != text_embeds.size(0):
                    raise ValueError(
                        "Contrastive loss requires the same batch size for images and texts."
                    )
                loss = self.contrastive_loss(
                    logits_per_image=logits_per_image,
                    logits_per_text=logits_per_text,
                )

        if return_dict:
            return CLIPContrastiveOutput(
                loss=loss,
                logits_per_image=logits_per_image,
                logits_per_text=logits_per_text,
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                logit_scale=logit_scale,
            )

        return (
            loss,
            logits_per_image,
            logits_per_text,
            image_embeds,
            text_embeds,
            logit_scale,
        )

    def _maybe_normalize(
        self,
        embeds: Tensor,
        normalize: Optional[bool] = None,
    ) -> Tensor:
        if normalize is None:
            normalize = self.normalize
        if not normalize:
            return embeds
        return F.normalize(embeds, dim=-1)
