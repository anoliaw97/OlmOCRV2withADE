import base64
import io
import subprocess
from typing import List

from PIL import Image


def get_pdf_media_box_width_height(local_pdf_path: str, page_num: int) -> tuple[float, float]:
    """
    Get the MediaBox dimensions for a specific page in a PDF file using the pdfinfo command.

    :param pdf_file: Path to the PDF file
    :param page_num: The page number for which to extract MediaBox dimensions
    :return: A tuple of (width, height) or raises ValueError
    """
    # Construct the pdfinfo command to extract info for the specific page
    command = ["pdfinfo", "-f", str(page_num), "-l", str(page_num), "-box", "-enc", "UTF-8", local_pdf_path]

    # Run the command using subprocess
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)

    # Check if there is any error in executing the command
    if result.returncode != 0:
        raise ValueError(f"pdfinfo failed (rc={result.returncode}): {result.stderr[:200]}")

    # Parse the output to find MediaBox
    output = result.stdout

    for line in output.splitlines():
        if "MediaBox" in line:
            parts = line.split(":", 1)  # Split only on first colon
            if len(parts) < 2:
                raise ValueError(f"MediaBox line has no colon: {line}")
            media_box_str = parts[1].strip().split()
            if len(media_box_str) < 4:
                raise ValueError(f"MediaBox line has insufficient values: {line}")
            media_box: List[float] = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])

    raise ValueError("MediaBox not found in the PDF info.")


def render_pdf_to_base64png(local_pdf_path: str, page_num: int, target_longest_image_dim: int = 2048) -> str:
    from pdf2image import convert_from_path
    import tempfile
    
    try:
        images = convert_from_path(
            local_pdf_path,
            first_page=page_num,
            last_page=page_num,
            dpi=150,
            fmt='png'
        )
        if not images:
            raise ValueError("pdf2image produced no output")
        
        img = images[0]
        longest_dim = max(img.size)
        scale = target_longest_image_dim / longest_dim if longest_dim > target_longest_image_dim else 1
        
        if scale < 1:
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"pdf2image failed: {e}")


def render_pdf_to_base64webp(local_pdf_path: str, page: int, target_longest_image_dim: int = 1024):
    base64_png = render_pdf_to_base64png(local_pdf_path, page, target_longest_image_dim)

    png_image = Image.open(io.BytesIO(base64.b64decode(base64_png)))
    webp_output = io.BytesIO()
    png_image.save(webp_output, format="WEBP")

    return base64.b64encode(webp_output.getvalue()).decode("utf-8")


def get_png_dimensions_from_base64(base64_data) -> tuple[int, int]:
    """
    Returns the (width, height) of a PNG image given its base64-encoded data,
    without base64-decoding the entire data or loading the PNG itself

    Should be really fast to support filtering

    Parameters:
    - base64_data (str): Base64-encoded PNG image data.

    Returns:
    - tuple: (width, height) of the image.

    Raises:
    - ValueError: If the data is not a valid PNG image or the required bytes are not found.
    """
    # PNG signature is 8 bytes
    png_signature_base64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
    if not base64_data.startswith(png_signature_base64[:8]):
        raise ValueError("Not a valid PNG file")

    # Positions in the binary data where width and height are stored
    width_start = 16  # Byte position where width starts (0-based indexing)
    _width_end = 20  # Byte position where width ends (exclusive)
    _height_start = 20
    height_end = 24

    # Compute the byte range needed (from width_start to height_end)
    start_byte = width_start
    end_byte = height_end

    # Calculate base64 character positions
    # Each group of 3 bytes corresponds to 4 base64 characters
    base64_start = (start_byte // 3) * 4
    base64_end = ((end_byte + 2) // 3) * 4  # Add 2 to ensure we cover partial groups

    # Extract the necessary base64 substring
    base64_substring = base64_data[base64_start:base64_end]

    # Decode only the necessary bytes
    decoded_bytes = base64.b64decode(base64_substring)

    # Compute the offset within the decoded bytes
    offset = start_byte % 3

    # Extract width and height bytes
    width_bytes = decoded_bytes[offset : offset + 4]
    height_bytes = decoded_bytes[offset + 4 : offset + 8]

    if len(width_bytes) < 4 or len(height_bytes) < 4:
        raise ValueError("Insufficient data to extract dimensions")

    # Convert bytes to integers
    width = int.from_bytes(width_bytes, "big")
    height = int.from_bytes(height_bytes, "big")

    return width, height
