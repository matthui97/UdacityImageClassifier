def process_image(image):
    image_shape = 224
    batch_size = 64
    processed_image = tf.image.resize(image, [image_shape,image_shape], preserve_aspect_ratio=False)
    return processed_image