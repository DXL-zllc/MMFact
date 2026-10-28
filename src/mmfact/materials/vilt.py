"""ViLT preprocessing with a valid patch grid for very narrow images."""


def prepare_vilt_pair(processor, image, text, max_length):
    import numpy as np
    from transformers.models.vilt.image_processing_vilt import (
        get_resize_output_image_size,
    )

    image = image.convert("RGB")
    options = {}
    pixels = processor.image_processor
    if pixels.do_resize:
        shorter, divisor = pixels.size["shortest_edge"], pixels.size_divisor
        height, width = get_resize_output_image_size(
            np.asarray(image),
            shorter=shorter,
            longer=int(1333 / 800 * shorter),
            size_divisor=divisor,
            input_data_format="channels_last",
        )
        if min(height, width) < divisor:
            image = image.resize(
                (max(width, divisor), max(height, divisor)), resample=pixels.resample
            )
            options["do_resize"] = False
    return processor(
        images=image,
        text=text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        input_data_format="channels_last",
        **options,
    )
