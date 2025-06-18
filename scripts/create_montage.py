from PIL import Image
import os

def create_montage_packed(all_image_paths,
                          output_path,
                          max_width=2000,
                          padding=10,
                          target_row_height=250,
                          bg_color="white"):
    """
    Build a densely-packed montage (Pinterest / justified-gallery style).

    Parameters
    ----------
    all_image_paths : list[str]   – paths to every image to include
    output_path      : str        – where to save the montage (PNG)
    max_width        : int        – final montage width in px, including padding
    padding          : int        – gap between thumbs and around the border
    target_row_height: int        – desired average thumb height before scaling
    bg_color         : str/tuple  – PIL color for background

    The algorithm fills each row so it spans the full width, then
    scales the row’s height accordingly; it keeps aspect ratio for every
    thumbnail, yielding minimal wasted space.  The last row is left-aligned
    at the target height (so it isn’t stretched unnaturally).
    """
    if not all_image_paths:
        raise ValueError("No images given")

    # ---------- Load all images (keep PIL objects so we can reuse) ----------
    images = []
    for p in all_image_paths:
        try:
            images.append(Image.open(p))
        except Exception as e:
            print(f"⚠️  Skipped {p}: {e}")

    if not images:
        raise RuntimeError("None of the images could be opened")

    # ---------- First pass: decide rows, sizes, positions ----------
    rows = []                # each item: list[dict(w, h, img)]
    row, row_ar_sum = [], 0  # working row and its Σ(w/h)

    # Helper: commit the current row once it's wide enough
    def _flush_row(force=False):
        nonlocal row, row_ar_sum
        if not row:
            return
        # For the last row (force=True) we keep target height
        row_w_available = max_width - padding * (len(row) + 1)
        if not force:
            # scale row height so total width (incl. gaps) == max_width
            row_height = row_w_available / row_ar_sum
        else:  # last row
            row_height = target_row_height
        # Record final size for every thumb
        for cell in row:
            w = int(row_height * cell["ar"])          # ar = w/h
            h = int(row_height)
            cell.update(final_w=w, final_h=h)
        rows.append(row)
        row, row_ar_sum = [], 0

    for img in images:
        ar = img.width / img.height
        row.append({"img": img, "ar": ar})
        row_ar_sum += ar

        # Predict height if we closed the row now
        row_w_available = max_width - padding * (len(row) + 1)
        row_height = row_w_available / row_ar_sum
        # If it would get too small, finish the row _without_ this image
        if row_height < target_row_height * 0.9 and len(row) > 1:
            # pop last image, flush row, start new row with that image
            last = row.pop()
            row_ar_sum -= last["ar"]
            _flush_row()
            row.append(last)
            row_ar_sum += last["ar"]

    # Flush whatever remains (last row – left-aligned)
    _flush_row(force=True)

    # ---------- Compute montage height ----------
    total_height = padding  # top margin
    for row in rows:
        row_h = max(c["final_h"] for c in row)
        total_height += row_h + padding        # each row + gap

    # ---------- Create canvas and paste ----------
    montage = Image.new("RGB", (max_width, total_height), color=bg_color)
    y_offset = padding
    for row in rows:
        x_offset = padding
        row_h = max(c["final_h"] for c in row)
        for cell in row:
            # Resize only once, keep good quality
            thumb = cell["img"].resize((cell["final_w"], cell["final_h"]),
                                       Image.Resampling.LANCZOS)
            montage.paste(thumb, (x_offset, y_offset + (row_h - cell["final_h"]) // 2))
            x_offset += cell["final_w"] + padding
        y_offset += row_h + padding

    # ---------- Save ----------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    montage.save(output_path, "PNG", quality=95)
    print(f"✅  Montage saved to {output_path}  "
          f"({montage.width} × {montage.height} px, {len(images)} images)")
