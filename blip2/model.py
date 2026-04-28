import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from transformers import Blip2ForConditionalGeneration, Blip2ForImageTextRetrieval
except ImportError as exc:  # pragma: no cover - depends on the local environment.
    Blip2ForConditionalGeneration = None
    Blip2ForImageTextRetrieval = None
    _BLIP2_IMPORT_ERROR = exc
else:
    _BLIP2_IMPORT_ERROR = None


def _gelu(x: Tensor) -> Tensor:
    return F.gelu(x)


@dataclass
class BLIP2Stage1Output:
    loss: Optional[Tensor]
    itc_loss: Optional[Tensor]
    itm_loss: Optional[Tensor]
    itg_loss: Optional[Tensor]
    logits_per_image: Optional[Tensor]
    logits_per_text: Optional[Tensor]
    image_embeds: Optional[Tensor]
    text_embeds: Optional[Tensor]
    itm_logits: Optional[Tensor]


@dataclass
class BLIP2Stage2Output:
    loss: Optional[Tensor]
    logits: Optional[Tensor]


class BLIP2QFormerLMHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def tie_weights(self, embedding_layer: nn.Embedding) -> None:
        self.decoder.weight = embedding_layer.weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = _gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return self.decoder(hidden_states)


class BLIP2Stage1Model(nn.Module):
    training_stage = "stage1"

    def __init__(
        self,
        blip2: "Blip2ForImageTextRetrieval",
        normalize: bool = True,
        itc_weight: float = 1.0,
        itm_weight: float = 1.0,
        itg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.blip2 = blip2
        self.normalize = normalize
        self.itc_weight = itc_weight
        self.itm_weight = itm_weight
        self.itg_weight = itg_weight
        self.itg_head = BLIP2QFormerLMHead(
            hidden_size=blip2.config.qformer_config.hidden_size,
            vocab_size=blip2.config.qformer_config.vocab_size,
            layer_norm_eps=blip2.config.qformer_config.layer_norm_eps,
        )
        self.itg_head.tie_weights(self.blip2.embeddings.word_embeddings)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Salesforce/blip2-itm-vit-g",
        normalize: bool = True,
        train_vision: bool = True,
        train_qformer: bool = True,
        train_text_embeddings: bool = True,
        train_projection: bool = True,
        itc_weight: float = 1.0,
        itm_weight: float = 1.0,
        itg_weight: float = 1.0,
        **kwargs,
    ) -> "BLIP2Stage1Model":
        if Blip2ForImageTextRetrieval is None:
            raise ImportError(
                "BLIP-2 support requires `transformers` with "
                "`Blip2ForImageTextRetrieval` available."
            ) from _BLIP2_IMPORT_ERROR

        model_path = Path(model_name_or_path)
        stage1_state_path = model_path / "stage1_state.pt"
        backbone_path = model_path / "backbone"

        if model_path.is_dir() and stage1_state_path.exists() and backbone_path.exists():
            blip2 = Blip2ForImageTextRetrieval.from_pretrained(str(backbone_path), **kwargs)
            state = torch.load(stage1_state_path, map_location="cpu")
            model = cls(
                blip2=blip2,
                normalize=bool(state.get("normalize", normalize)),
                itc_weight=float(state.get("itc_weight", itc_weight)),
                itm_weight=float(state.get("itm_weight", itm_weight)),
                itg_weight=float(state.get("itg_weight", itg_weight)),
            )
            model.load_state_dict(state["model_state_dict"])
        else:
            blip2 = Blip2ForImageTextRetrieval.from_pretrained(model_name_or_path, **kwargs)
            model = cls(
                blip2=blip2,
                normalize=normalize,
                itc_weight=itc_weight,
                itm_weight=itm_weight,
                itg_weight=itg_weight,
            )

        model.set_trainable(
            train_vision=train_vision,
            train_qformer=train_qformer,
            train_text_embeddings=train_text_embeddings,
            train_projection=train_projection,
        )
        return model

    def save_pretrained(self, save_directory: str | PathLike[str]) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.blip2.save_pretrained(save_path / "backbone")
        metadata = {
            "training_stage": self.training_stage,
            "normalize": self.normalize,
            "itc_weight": self.itc_weight,
            "itm_weight": self.itm_weight,
            "itg_weight": self.itg_weight,
        }
        (save_path / "stage1_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        torch.save(
            {
                **metadata,
                "model_state_dict": self.state_dict(),
            },
            save_path / "stage1_state.pt",
        )

    def set_trainable(
        self,
        train_vision: bool = True,
        train_qformer: bool = True,
        train_text_embeddings: bool = True,
        train_projection: bool = True,
    ) -> None:
        self._set_module_grad(self.blip2.vision_model, train_vision)
        self._set_module_grad(self.blip2.qformer, train_qformer)
        self._set_module_grad(self.blip2.embeddings, train_text_embeddings)
        self._set_module_grad(self.blip2.vision_projection, train_projection)
        self._set_module_grad(self.blip2.text_projection, train_projection)
        self._set_module_grad(self.blip2.itm_head, train_projection)
        self._set_module_grad(self.itg_head, train_qformer)
        self.blip2.query_tokens.requires_grad = train_qformer

    @staticmethod
    def _set_module_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _build_image_attention_mask(self, image_hidden_states: Tensor) -> Tensor:
        return torch.ones(
            image_hidden_states.shape[:-1],
            dtype=torch.long,
            device=image_hidden_states.device,
        )

    def _expand_query_tokens(self, batch_size: int) -> Tensor:
        return self.blip2.query_tokens.expand(batch_size, -1, -1)

    def encode_image(
        self,
        pixel_values: Tensor,
        normalize: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Tensor:
        image_hidden_states = self._encode_vision_hidden_states(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_attention_mask = self._build_image_attention_mask(image_hidden_states)
        return self.encode_image_from_hidden_states(
            image_hidden_states=image_hidden_states,
            image_attention_mask=image_attention_mask,
            normalize=normalize,
        )

    def encode_image_from_hidden_states(
        self,
        image_hidden_states: Tensor,
        image_attention_mask: Tensor,
        normalize: Optional[bool] = None,
    ) -> Tensor:
        query_tokens = self._expand_query_tokens(image_hidden_states.size(0))
        qformer_outputs = self.blip2.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_hidden_states,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_states = self._coerce_last_hidden_state(qformer_outputs, "qformer")
        query_states = query_states.to(dtype=self.blip2.vision_projection.weight.dtype)
        image_embeds = self.blip2.vision_projection(query_states)
        return self._maybe_normalize(image_embeds, normalize)

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        normalize: Optional[bool] = None,
    ) -> Tensor:
        del position_ids

        text_input_ids, text_attention_mask = self._prepare_text_only_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeddings = self.blip2.embeddings(input_ids=text_input_ids)
        qformer_kwargs = {
            "query_embeds": text_embeddings,
            "attention_mask": text_attention_mask,
            "return_dict": True,
        }

        try:
            qformer_outputs = self.blip2.qformer(query_length=0, **qformer_kwargs)
        except TypeError:
            qformer_outputs = self.blip2.qformer(**qformer_kwargs)

        text_hidden_states = self._coerce_last_hidden_state(qformer_outputs, "text")
        text_hidden_states = text_hidden_states.to(dtype=self.blip2.text_projection.weight.dtype)
        text_embeds = self.blip2.text_projection(text_hidden_states[:, 0, :])
        return self._maybe_normalize(text_embeds, normalize)

    def compute_similarity(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = torch.matmul(image_embeds, text_embeds.t())
        if logits.ndim == 3:
            logits_per_image = logits.max(dim=1).values
        elif logits.ndim == 2:
            logits_per_image = logits
        else:
            raise TypeError(
                f"Unsupported BLIP-2 image/text similarity shape: {tuple(logits.shape)}."
            )
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

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
    ) -> BLIP2Stage1Output | tuple[
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        del position_ids

        image_embeds = None
        text_embeds = None
        logits_per_image = None
        logits_per_text = None
        total_loss = None
        itc_loss = None
        itm_loss = None
        itg_loss = None
        itm_logits = None

        image_hidden_states = None
        image_attention_mask = None
        if pixel_values is not None:
            image_hidden_states = self._encode_vision_hidden_states(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
            image_attention_mask = self._build_image_attention_mask(image_hidden_states)
            image_embeds = self.encode_image_from_hidden_states(
                image_hidden_states=image_hidden_states,
                image_attention_mask=image_attention_mask,
            )

        if input_ids is not None:
            text_embeds = self.encode_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if image_embeds is not None and text_embeds is not None:
            logits_per_image, logits_per_text = self.compute_similarity(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
            )

            if return_loss:
                if image_embeds.size(0) != text_embeds.size(0):
                    raise ValueError(
                        "Contrastive loss requires the same batch size for images and texts."
                    )

                losses = []
                if self.itc_weight != 0.0:
                    itc_loss = self.contrastive_loss(
                        logits_per_image=logits_per_image,
                        logits_per_text=logits_per_text,
                    )
                    losses.append(self.itc_weight * itc_loss)

                if self.itm_weight != 0.0 and image_hidden_states is not None:
                    itm_loss, itm_logits = self.compute_itm_loss(
                        image_hidden_states=image_hidden_states,
                        image_attention_mask=image_attention_mask,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        logits_per_image=logits_per_image,
                        logits_per_text=logits_per_text,
                    )
                    losses.append(self.itm_weight * itm_loss)

                if self.itg_weight != 0.0 and image_hidden_states is not None:
                    itg_loss = self.compute_itg_loss(
                        image_hidden_states=image_hidden_states,
                        image_attention_mask=image_attention_mask,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    losses.append(self.itg_weight * itg_loss)

                if losses:
                    total_loss = torch.stack(losses).sum()
                else:
                    total_loss = logits_per_image.new_zeros(())

        if return_dict:
            return BLIP2Stage1Output(
                loss=total_loss,
                itc_loss=itc_loss,
                itm_loss=itm_loss,
                itg_loss=itg_loss,
                logits_per_image=logits_per_image,
                logits_per_text=logits_per_text,
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                itm_logits=itm_logits,
            )

        return (
            total_loss,
            logits_per_image,
            logits_per_text,
            image_embeds,
            text_embeds,
        )

    def compute_itm_loss(
        self,
        image_hidden_states: Tensor,
        image_attention_mask: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        logits_per_image: Tensor,
        logits_per_text: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = input_ids.size(0)
        if batch_size < 2:
            zero = image_hidden_states.new_zeros(())
            empty_logits = image_hidden_states.new_zeros((0, 2))
            return zero, empty_logits

        negative_text_indices = self._mine_hard_negative_indices(logits_per_image.detach())
        negative_image_indices = self._mine_hard_negative_indices(logits_per_text.detach())

        positive_logits = self._compute_itm_logits(
            image_hidden_states=image_hidden_states,
            image_attention_mask=image_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        negative_text_logits = self._compute_itm_logits(
            image_hidden_states=image_hidden_states,
            image_attention_mask=image_attention_mask,
            input_ids=input_ids[negative_text_indices],
            attention_mask=None if attention_mask is None else attention_mask[negative_text_indices],
        )
        negative_image_logits = self._compute_itm_logits(
            image_hidden_states=image_hidden_states[negative_image_indices],
            image_attention_mask=image_attention_mask[negative_image_indices],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        itm_logits = torch.cat(
            [positive_logits, negative_text_logits, negative_image_logits],
            dim=0,
        )
        itm_labels = torch.cat(
            [
                torch.ones(batch_size, device=itm_logits.device, dtype=torch.long),
                torch.zeros(2 * batch_size, device=itm_logits.device, dtype=torch.long),
            ],
            dim=0,
        )
        itm_loss = F.cross_entropy(itm_logits, itm_labels)
        return itm_loss, itm_logits

    def _compute_itm_logits(
        self,
        image_hidden_states: Tensor,
        image_attention_mask: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        query_tokens = self._expand_query_tokens(image_hidden_states.size(0))
        query_text_embeds, qformer_attention_mask = self._prepare_query_text_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            query_tokens=query_tokens,
        )
        text_outputs = self.blip2.qformer(
            query_embeds=query_text_embeds,
            query_length=query_tokens.size(1),
            attention_mask=qformer_attention_mask,
            encoder_hidden_states=image_hidden_states,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        joint_hidden_states = self._coerce_last_hidden_state(text_outputs, "itm")
        joint_hidden_states = joint_hidden_states.to(dtype=self.blip2.itm_head.weight.dtype)
        output = self.blip2.itm_head(joint_hidden_states[:, : query_tokens.size(1), :])
        return output.mean(dim=1)

    @staticmethod
    def _mine_hard_negative_indices(similarity: Tensor) -> Tensor:
        masked_similarity = similarity.clone()
        diagonal = torch.eye(
            masked_similarity.size(0),
            device=masked_similarity.device,
            dtype=torch.bool,
        )
        masked_similarity = masked_similarity.masked_fill(diagonal, torch.finfo(masked_similarity.dtype).min)
        return masked_similarity.argmax(dim=1)

    def compute_itg_loss(
        self,
        image_hidden_states: Tensor,
        image_attention_mask: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        text_input_ids, text_attention_mask = self._prepare_text_only_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if text_input_ids.size(1) < 2:
            return image_hidden_states.new_zeros(())

        query_tokens = self._expand_query_tokens(text_input_ids.size(0))
        query_text_embeds = self.blip2.embeddings(
            input_ids=text_input_ids,
            query_embeds=query_tokens,
        )
        qformer_attention_mask = self._build_itg_attention_mask(
            text_attention_mask=text_attention_mask,
            query_length=query_tokens.size(1),
        )
        qformer_outputs = self.blip2.qformer(
            query_embeds=query_text_embeds,
            query_length=query_tokens.size(1),
            attention_mask=qformer_attention_mask,
            encoder_hidden_states=image_hidden_states,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        hidden_states = self._coerce_last_hidden_state(qformer_outputs, "itg")
        text_hidden_states = hidden_states[:, query_tokens.size(1) :, :]
        text_hidden_states = text_hidden_states.to(dtype=self.itg_head.decoder.weight.dtype)
        lm_logits = self.itg_head(text_hidden_states)

        labels = text_input_ids.clone()
        labels[text_attention_mask == 0] = -100
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        valid_targets = bool((shift_labels != -100).any().item())
        if not valid_targets:
            return image_hidden_states.new_zeros(())

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _build_itg_attention_mask(
        text_attention_mask: Tensor,
        query_length: int,
    ) -> Tensor:
        batch_size, text_length = text_attention_mask.shape
        device = text_attention_mask.device
        dtype = text_attention_mask.dtype

        total_length = query_length + text_length
        attention_mask = torch.zeros(
            batch_size,
            total_length,
            total_length,
            device=device,
            dtype=dtype,
        )

        attention_mask[:, :query_length, :query_length] = 1
        attention_mask[:, query_length:, :query_length] = 1

        causal_text_mask = torch.tril(
            torch.ones(text_length, text_length, device=device, dtype=dtype)
        )
        attention_mask[:, query_length:, query_length:] = causal_text_mask

        key_mask = text_attention_mask[:, None, :].to(dtype)
        attention_mask[:, :, query_length:] = attention_mask[:, :, query_length:] * key_mask
        return attention_mask

    def _encode_vision_hidden_states(
        self,
        pixel_values: Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> Tensor:
        vision_outputs = self._run_vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        return self._coerce_last_hidden_state(vision_outputs, "vision")

    def _prepare_text_only_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        image_token_index = getattr(self.blip2.config, "image_token_index", None)
        if image_token_index is not None:
            num_query_tokens = getattr(self.blip2.config, "num_query_tokens", 0)
            if num_query_tokens > 0 and input_ids.size(1) > num_query_tokens:
                input_ids = input_ids[:, num_query_tokens:]
                attention_mask = attention_mask[:, num_query_tokens:]
        return input_ids, attention_mask

    def _prepare_query_text_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        query_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        image_token_index = getattr(self.blip2.config, "image_token_index", None)
        if image_token_index is not None:
            num_query_tokens = getattr(self.blip2.config, "num_query_tokens", 0)
            if num_query_tokens > 0 and input_ids.size(1) > num_query_tokens:
                input_ids = input_ids[:, num_query_tokens:]
        else:
            query_attention_mask = torch.ones(
                query_tokens.size()[:-1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)

        query_text_embeds = self.blip2.embeddings(
            input_ids=input_ids,
            query_embeds=query_tokens,
        )
        return query_text_embeds, attention_mask

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

    def _run_vision_model(
        self,
        pixel_values: Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> Tensor | tuple | object:
        try:
            return self.blip2.vision_model(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=True,
            )
        except TypeError:
            return self.blip2.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )

    @staticmethod
    def _coerce_last_hidden_state(
        outputs: Tensor | tuple | object,
        modality: str,
    ) -> Tensor:
        if torch.is_tensor(outputs):
            return outputs

        hidden_state = getattr(outputs, "last_hidden_state", None)
        if torch.is_tensor(hidden_state):
            return hidden_state

        if isinstance(outputs, (tuple, list)):
            for item in outputs:
                if torch.is_tensor(item) and item.ndim >= 2:
                    return item

        raise TypeError(
            f"Unsupported BLIP-2 {modality} output type: {type(outputs)!r}. "
            "Expected a tensor or an object containing last_hidden_state."
        )


class BLIP2Stage2Model(nn.Module):
    training_stage = "stage2"

    def __init__(self, blip2: "Blip2ForConditionalGeneration") -> None:
        super().__init__()
        self.blip2 = blip2

    @property
    def use_decoder_only_language_model(self) -> bool:
        return bool(getattr(self.blip2.config, "use_decoder_only_language_model", False))

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Salesforce/blip2-flan-t5-xl",
        train_vision: bool = False,
        train_qformer: bool = True,
        train_language_model: bool = False,
        train_language_projection: bool = True,
        **kwargs,
    ) -> "BLIP2Stage2Model":
        if Blip2ForConditionalGeneration is None:
            raise ImportError(
                "BLIP-2 stage-2 support requires `transformers` with "
                "`Blip2ForConditionalGeneration` available."
            ) from _BLIP2_IMPORT_ERROR

        blip2 = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        model = cls(blip2=blip2)
        model.set_trainable(
            train_vision=train_vision,
            train_qformer=train_qformer,
            train_language_model=train_language_model,
            train_language_projection=train_language_projection,
        )
        return model

    def save_pretrained(self, save_directory: str | PathLike[str]) -> None:
        self.blip2.save_pretrained(save_directory)

    def set_trainable(
        self,
        train_vision: bool = False,
        train_qformer: bool = True,
        train_language_model: bool = False,
        train_language_projection: bool = True,
    ) -> None:
        self._set_module_grad(self.blip2.vision_model, train_vision)
        self._set_module_grad(self.blip2.qformer, train_qformer)
        self._set_module_grad(self.blip2.language_model, train_language_model)
        self._set_module_grad(self.blip2.language_projection, train_language_projection)
        self.blip2.query_tokens.requires_grad = train_qformer

    @staticmethod
    def _set_module_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> BLIP2Stage2Output | tuple[Optional[Tensor], Optional[Tensor]]:
        outputs = self.blip2(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
            **kwargs,
        )

        if return_dict:
            return BLIP2Stage2Output(
                loss=outputs.loss,
                logits=outputs.logits,
            )
        return outputs.loss, outputs.logits
