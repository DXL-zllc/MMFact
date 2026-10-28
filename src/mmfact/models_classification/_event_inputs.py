"""Content-bound CLIP tokens, event spans and detected crops for Event-Radar."""

import torch
from PIL import Image
from torch.nn import functional as F

from mmfact.data.features import file_digest, fingerprint
from mmfact.models_classification._encoders import CLIPImageEncoder, CLIPTextEncoder

_ROLES = ("subject", "object", "location", "predicate")


def _span_mask(text, span, offsets, valid):
    """Partially truncated or uncovered spans are missing, never partial entities."""
    selected = torch.zeros_like(valid)
    if span is None:
        return selected
    start, end = span["start"], span["end"]
    if text[start:end] != span["text"]:
        raise ValueError("Event-Radar role offsets do not match the original text.")
    selected = valid & (offsets[:, 0] < end) & (offsets[:, 1] > start)
    covered = set()
    for left, right in offsets[selected].tolist():
        covered.update(range(max(left, start), min(right, end)))
    if any(index not in covered and not text[index].isspace() for index in range(start, end)):
        selected.zero_()
    return selected


class _EventText(CLIPTextEncoder):
    """Cache raw projected tokens unchanged; aggregate roles after the cache."""

    def __init__(self, **options):
        super().__init__(**options)
        if not self.tokenizer.is_fast:
            raise ValueError("Event-Radar needs a fast tokenizer with original-text offsets.")

    def event_features(self, texts, records):
        tokens = self(texts)
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        ordinary = self._tokenize(texts)
        if any(
            not torch.equal(encoded[key], ordinary[key]) for key in ("input_ids", "attention_mask")
        ):
            raise ValueError("Event-Radar span tokenizer differs from the actual CLIP input.")
        context_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
        if self._cls_pooling:
            global_features = tokens[:, 0]
        else:
            # Preserve HF's legacy EOS=2 pooling as well as its current first-EOS rule.
            eos = (
                self.tokenizer.eos_token_id
                if hasattr(self, "matrix")
                else self.network["encoder"].config.eos_token_id
            )
            ids = encoded["input_ids"]
            indices = ids.argmax(-1) if eos == 2 else ids.eq(eos).int().argmax(-1)
            global_features = tokens[
                torch.arange(len(texts), device=tokens.device), indices.to(tokens.device)
            ]
        roles, masks = [], []
        for text, record, features, offsets, valid in zip(
            texts, records, tokens, encoded["offset_mapping"], context_mask
        ):
            selections = torch.stack(
                [_span_mask(text, record["roles"][role], offsets, valid) for role in _ROLES]
            ).to(tokens.device)
            weights = selections.to(tokens.dtype)
            roles.append(weights @ features / weights.sum(-1, keepdim=True).clamp_min(1))
            masks.append(selections.any(-1))
        roles, masks = torch.stack(roles), torch.stack(masks)
        enhanced = torch.where(
            masks[:, 3, None], (global_features + roles[:, 3]) / 2, global_features
        )
        event = torch.cat((enhanced[:, None], roles[:, :3]), 1)
        event_mask = torch.cat((torch.ones_like(masks[:, :1]), masks[:, :3]), 1)
        return event, event_mask, tokens, context_mask.to(tokens.device)


class _EventImage(CLIPImageEncoder):
    """One frozen tower for the global image and valid, precomputed object crops."""

    def __init__(self, resources, encoding_batch_size=32, **options):
        super().__init__(**options)
        if type(encoding_batch_size) is not int or encoding_batch_size < 1:
            raise ValueError("Event-Radar image encoding_batch_size must be positive.")
        self._objects = resources.image_objects
        self.max_objects, self.encoding_batch_size = resources.max_objects, encoding_batch_size
        self.encoding_spec.update(
            region_recipe="event_radar_original_pixel_boxes_v1",
            max_objects=self.max_objects,
            padding="zero_features_after_encoding_valid_crops",
        )

    def inputs_from_rows(self, rows):
        paths = [row["image_path"] for row in rows]
        actual = {file_digest(path): self._objects(path) for path in dict.fromkeys(paths)}
        self.encoding_spec["object_records_sha256"] = fingerprint(actual)
        return paths

    def _encode(self, paths):
        counts, pixels = [], []
        for path in paths:
            record = self._objects(path)
            with Image.open(path) as image:
                image = image.convert("RGB")
                if list(image.size) != record["image_size"]:
                    raise ValueError("Event-Radar object coordinates have a different image size.")
                crops = [image, *(image.crop(obj["box"]) for obj in record["objects"])]
                values = self.processor(images=crops, return_tensors="pt")["pixel_values"]
            counts.append(len(crops))
            pixels.append(values)
        features = torch.cat(
            [
                self._project({"pixel_values": chunk.to(self.device)})
                for chunk in torch.cat(pixels).split(self.encoding_batch_size)
            ]
        )
        return torch.stack(
            [
                F.pad(value, (0, 0, 0, 1 + self.max_objects - count))
                for value, count in zip(features.split(counts), counts)
            ]
        )


def _event_nodes(text_encoder, image_encoder, texts, paths, records):
    text_event, text_mask, tokens, token_mask = text_encoder.event_features(texts, records)
    image = image_encoder(paths)
    batch, _, dimension = tokens.shape
    image_offset = 4 + tokens.shape[1]
    image_mask = (
        torch.arange(image.shape[1], device=image.device)[None]
        <= torch.tensor([len(record["objects"]) for record in records], device=image.device)[
            :, None
        ]
    )
    text_indices = torch.arange(4, device=tokens.device)[None].expand(batch, -1)
    image_indices = torch.full_like(text_indices, image_offset)
    image_role_mask = torch.ones_like(text_mask)
    for index, record in enumerate(records):
        count = len(record["objects"])
        image_role_mask[index, 1:] = text_mask[index, 1:] & (count > 0)
        if count:
            similarities = (
                F.normalize(text_event[index, 1:], dim=-1)
                @ F.normalize(image[index, 1 : count + 1], dim=-1).T
            )
            image_indices[index, 1:] += 1 + similarities.argmax(-1)
    nodes = torch.cat((text_event, tokens, image), 1)
    mask = torch.cat((text_mask, token_mask, image_mask), 1)
    if nodes.shape != (batch, mask.shape[1], dimension):
        raise ValueError("Event-Radar CLIP node dimensions differ.")
    return nodes, mask, text_indices, image_indices, text_mask, image_role_mask
