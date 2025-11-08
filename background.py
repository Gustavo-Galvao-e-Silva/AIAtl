import os
import random
from rembg import remove
from PIL import Image


def _load_image(img_class: str, img_name: str) -> Image.Image:
    """Load an image from the raw images directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    raw_images_dir = os.path.join(project_root, "data", "raw", img_class)
    img_path = os.path.join(raw_images_dir, img_name)

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    return Image.open(img_path).convert("RGBA")


def _remove_background(img: Image.Image) -> Image.Image:
    """Remove background from an image."""
    no_bg_image = remove(img).convert("RGBA")
    img.close()
    return no_bg_image


def _get_random_background() -> Image.Image:
    """Load a random background image."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    backgrounds_dir = os.path.join(project_root, "data", "backgrounds")

    background_file = random.choice(os.listdir(backgrounds_dir))
    background_path = os.path.join(backgrounds_dir, background_file)
    return Image.open(background_path).convert("RGBA")


def _composite_images(foreground: Image.Image, background: Image.Image) -> Image.Image:
    """Composite foreground onto background at a random position."""
    bg_width, bg_height = background.size
    fg_width, fg_height = foreground.size

    # Resize background if it's smaller than foreground
    if bg_width < fg_width or bg_height < fg_height:
        background = background.resize((fg_width, fg_height))
        bg_width, bg_height = background.size

    # Calculate random position
    max_x = bg_width - fg_width
    max_y = bg_height - fg_height
    x_off = random.randint(0, max_x)
    y_off = random.randint(0, max_y)

    # Create composite
    composite = background.copy()
    composite.alpha_composite(foreground, (x_off, y_off))

    return composite


def _save_image(img: Image.Image, img_class: str, base_name: str, index: int, output_dir: str):
    """Save composite image to the output directory."""
    class_output_dir = os.path.join(output_dir, img_class)
    os.makedirs(class_output_dir, exist_ok=True)

    output_filename = f"{base_name}_new_bg_{index}.jpg"
    output_path = os.path.join(class_output_dir, output_filename)

    img.convert("RGB").save(output_path, "JPEG")


def _image_add_new_backgrounds(img_class: str, img_name: str, num_backgrounds: int, output_dir: str):
    """Generate multiple images with new backgrounds."""
    # Load and process the original image
    img = _load_image(img_class, img_name)
    base_name = os.path.splitext(img_name)[0]

    _save_image(img, img_class, base_name, 0, output_dir)

    no_bg_image = _remove_background(img)

    # Generate composites with random backgrounds
    for i in range(1, num_backgrounds + 1):
        background_image = _get_random_background()
        composite = _composite_images(no_bg_image, background_image)

        _save_image(composite, img_class, base_name, i, output_dir)

        background_image.close()
        composite.close()

    no_bg_image.close()


def class_add_new_backgrounds(img_class: str, num_backgrounds: int):
    """Generate new backgrounds for all images in a class directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    raw_images_dir = os.path.join(project_root, "data", "raw", img_class)
    output_dir = os.path.join(project_root, "data", "new_bg")

    # Check if the class directory exists
    if not os.path.isdir(raw_images_dir):
        raise FileNotFoundError(f"Class directory not found: {raw_images_dir}")

    # Get all image files in the directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = os.listdir(raw_images_dir)
    if not image_files:
        print(f"No images found in {raw_images_dir}")
        return

    print(f"Processing {len(image_files)} images from class '{img_class}'...")

    # Process each image
    for idx, img_name in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing {img_name}...")
        try:
            _image_add_new_backgrounds(img_class, img_name, num_backgrounds, output_dir)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    print(f"Completed processing {len(image_files)} images!")


class_add_new_backgrounds("plastic", 2)